# SPDX-License-Identifier: Apache-2.0
"""Log rotation backstop and transient-artifact reaper (design doc R4).

Only `server.log` ever rotated on its own (`TimedRotatingFileHandler`
deletes old backups only at an actual midnight rollover — a restart-heavy
usage pattern can accumulate dated files past `backupCount`). The ad-hoc
log class (`fork-*.log`, `crash.log`, `installed-*.log`, `watchdog.log`)
is written by shell redirects and a watchdog script that appends forever,
with no rotation and no retention at all. Nothing bounded the directory as
a whole, and nothing ever swept the HF download staging class this module
also covers.

Every deletion target here is an allowlisted, age-gated transient pattern
— this is a report-and-reap routine, not a general-purpose cleaner. User
artifacts anywhere in the tree are never touched (design doc §E3/§9).
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# §7 rule 2: age gates on anything name-pattern-based.
_AD_HOC_LOG_MAX_AGE_DAYS = 14
_WATCHDOG_LOG_TRUNCATE_ABOVE_BYTES = 5 * 1024 * 1024
_WATCHDOG_LOG_TRUNCATE_KEEP_BYTES = 1 * 1024 * 1024
_LOG_DIR_CAP_BYTES = 500 * 1024 * 1024
_HF_STAGING_MAX_AGE_DAYS = 7
_SERVER_LOG_BACKUP_COUNT = 7

_DATED_SERVER_LOG_RE = re.compile(r"^server\.log\.\d{4}-\d{2}-\d{2}$")
_AD_HOC_LOG_GLOBS = ("fork-*.log", "crash.log", "installed-*.log")


@dataclass
class LogArtifactReapResult:
    dated_server_logs_pruned: int = 0
    ad_hoc_logs_deleted: int = 0
    watchdog_log_truncated: bool = False
    cap_evicted_count: int = 0
    cap_evicted_bytes: int = 0
    hf_staging_removed: int = 0
    pycache_removed: bool = False
    errors: list[str] = field(default_factory=list)


def _safe_unlink(path: Path, result: LogArtifactReapResult) -> bool:
    if path.is_symlink():
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        result.errors.append(f"unlink {path}: {exc}")
        return False


def _safe_rmtree(path: Path, result: LogArtifactReapResult) -> bool:
    if path.is_symlink():
        return False
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        result.errors.append(f"rmtree {path}: {exc}")
        return False


def _reap_dated_server_logs(log_dir: Path, result: LogArtifactReapResult) -> None:
    """Keep the newest `_SERVER_LOG_BACKUP_COUNT` dated rotations.

    Backstop for TimedRotatingFileHandler's restart gap: it only prunes at
    an actual midnight rollover, so a restart-heavy pattern can accumulate
    dated files beyond `backupCount` (design doc §E1).
    """
    dated = [p for p in log_dir.glob("server.log.*") if _DATED_SERVER_LOG_RE.match(p.name)]
    if len(dated) <= _SERVER_LOG_BACKUP_COUNT:
        return
    dated.sort(key=lambda p: p.name)  # ISO dates sort chronologically as strings
    for path in dated[: len(dated) - _SERVER_LOG_BACKUP_COUNT]:
        if _safe_unlink(path, result):
            result.dated_server_logs_pruned += 1


def _reap_ad_hoc_logs(log_dir: Path, result: LogArtifactReapResult) -> None:
    """Delete the unbounded ad-hoc log class once it's old enough."""
    now = time.time()
    for pattern in _AD_HOC_LOG_GLOBS:
        for path in log_dir.glob(pattern):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                age_days = (now - path.stat().st_mtime) / 86400
            except OSError:
                continue
            if age_days < _AD_HOC_LOG_MAX_AGE_DAYS:
                continue
            if _safe_unlink(path, result):
                result.ad_hoc_logs_deleted += 1


def _truncate_watchdog_log(log_dir: Path, result: LogArtifactReapResult) -> None:
    """A watchdog script appends forever with no rotation of its own."""
    path = log_dir / "watchdog.log"
    if path.is_symlink() or not path.is_file():
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= _WATCHDOG_LOG_TRUNCATE_ABOVE_BYTES:
        return
    try:
        with path.open("rb") as f:
            f.seek(-_WATCHDOG_LOG_TRUNCATE_KEEP_BYTES, 2)
            tail = f.read()
        with path.open("wb") as f:
            f.write(tail)
        result.watchdog_log_truncated = True
    except OSError as exc:
        result.errors.append(f"truncate {path}: {exc}")


def _cap_log_dir_total(log_dir: Path, result: LogArtifactReapResult) -> None:
    """Directory-wide cap, oldest-first-by-mtime, never the live server.log."""
    entries = []
    total = 0
    for path in log_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        total += st.st_size
        if path.name != "server.log":
            entries.append((st.st_mtime, path, st.st_size))

    if total <= _LOG_DIR_CAP_BYTES:
        return

    entries.sort(key=lambda e: e[0])  # oldest first
    for _mtime, path, size in entries:
        if total <= _LOG_DIR_CAP_BYTES:
            break
        if _safe_unlink(path, result):
            total -= size
            result.cap_evicted_count += 1
            result.cap_evicted_bytes += size


def _reap_hf_download_staging(models_dir: Path, result: LogArtifactReapResult) -> None:
    """Age-gated only — no active-download cross-reference (see module
    limitations note). A directory this old is not a download in progress:
    an active transfer touches its staging files continuously."""
    if not models_dir.is_dir():
        return
    now = time.time()
    for temp_dir in models_dir.glob("**/.cache/huggingface/download/._____temp"):
        if temp_dir.is_symlink() or not temp_dir.is_dir():
            continue
        try:
            age_days = (now - temp_dir.stat().st_mtime) / 86400
        except OSError:
            continue
        if age_days < _HF_STAGING_MAX_AGE_DAYS:
            continue
        if _safe_rmtree(temp_dir, result):
            result.hf_staging_removed += 1

    for lock_path in models_dir.glob("**/.cache/**/*.lock"):
        if lock_path.is_symlink() or not lock_path.is_file():
            continue
        try:
            age_days = (now - lock_path.stat().st_mtime) / 86400
        except OSError:
            continue
        if age_days < _HF_STAGING_MAX_AGE_DAYS:
            continue
        if _safe_unlink(lock_path, result):
            result.hf_staging_removed += 1


def _reap_pycache(base_path: Path, result: LogArtifactReapResult) -> None:
    """Regenerable; the fork script already sets PYTHONDONTWRITEBYTECODE=1."""
    pycache_dir = base_path / "__pycache__"
    if pycache_dir.is_symlink() or not pycache_dir.is_dir():
        return
    if _safe_rmtree(pycache_dir, result):
        result.pycache_removed = True


def run_log_artifact_reaper(
    *, base_path: Path, log_dir: Path, models_dir: Path | None = None
) -> LogArtifactReapResult:
    """One pass of the log/artifact reaper. Trigger: server startup + the
    disk-pressure guard tick (design doc §R4) — cheap, one directory
    listing per pattern.
    """
    result = LogArtifactReapResult()
    if log_dir.is_dir() and not log_dir.is_symlink():
        _reap_dated_server_logs(log_dir, result)
        _reap_ad_hoc_logs(log_dir, result)
        _truncate_watchdog_log(log_dir, result)
        _cap_log_dir_total(log_dir, result)
    if models_dir is not None:
        _reap_hf_download_staging(models_dir, result)
    _reap_pycache(base_path, result)

    if result.errors:
        for err in result.errors:
            logger.warning("Log/artifact reaper: %s", err)
    if (
        result.dated_server_logs_pruned
        or result.ad_hoc_logs_deleted
        or result.watchdog_log_truncated
        or result.cap_evicted_count
        or result.hf_staging_removed
        or result.pycache_removed
    ):
        logger.info(
            "Log/artifact reaper: dated_server_logs=%d ad_hoc_logs=%d "
            "watchdog_truncated=%s cap_evicted=%d (%d bytes) hf_staging=%d "
            "pycache=%s",
            result.dated_server_logs_pruned,
            result.ad_hoc_logs_deleted,
            result.watchdog_log_truncated,
            result.cap_evicted_count,
            result.cap_evicted_bytes,
            result.hf_staging_removed,
            result.pycache_removed,
        )
    return result
