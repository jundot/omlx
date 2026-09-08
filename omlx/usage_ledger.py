# SPDX-License-Identifier: Apache-2.0
"""
Per-session token usage ledger and cloud API cost estimation.

Complements :mod:`omlx.server_metrics`: where ``ServerMetrics`` tracks the
*current* session plus a persisted all-time aggregate, this module records
one closed-session snapshot per server run (a JSONL "ledger") and estimates
what the same token volumes would have cost against common paid cloud APIs,
so an operator can compare "served locally for $0" against "would have
cost $X on GPT-4o / Claude / Gemini".
"""

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import resolve_default_base_path

logger = logging.getLogger(__name__)

# Sessions retained in the JSONL ledger file. Oldest records are dropped
# once the file grows past this so a long-lived install doesn't accumulate
# an unbounded file across years of restarts.
MAX_LEDGER_SESSIONS = 1000

# Bundled defaults: cloud API list prices in USD per 1M tokens, keyed by a
# lowercase substring matched against the served model_id (see
# `PricingTable.lookup`). Figures are list prices for common hosted models
# as a rough basis for comparison — not a live pricing feed, and
# intentionally approximate. Shipped as package data (not a Python dict) so
# a user can override or extend the table without touching code — see
# `PricingTable`.
DEFAULT_PRICING_FILE = (
    Path(__file__).resolve().parent / "admin" / "data" / "cloud_pricing_defaults.json"
)

# User-editable overrides file name, stored under the resolved base path
# (same root as settings.json).
USER_PRICING_FILENAME = "cloud_pricing.json"


def _load_default_pricing_rows() -> List[Dict[str, Any]]:
    """Load the bundled default pricing rows from package data.

    Tolerant: a missing or corrupt file logs a warning and yields an empty
    table rather than raising, so a packaging mishap degrades to "no
    pricing known" instead of crashing the server.
    """
    try:
        with open(DEFAULT_PRICING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to load bundled cloud pricing defaults from %s: %s",
            DEFAULT_PRICING_FILE,
            e,
        )
        return []
    if not isinstance(data, list):
        logger.warning(
            "Bundled cloud pricing defaults at %s must be a JSON array; ignoring",
            DEFAULT_PRICING_FILE,
        )
        return []
    rows = []
    for row in data:
        try:
            rows.append(_validate_pricing_row(row))
        except ValueError as e:
            logger.warning(
                "Skipping invalid bundled pricing row in %s: %s",
                DEFAULT_PRICING_FILE,
                e,
            )
    return rows


def _validate_pricing_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a pricing row. Raises ``ValueError`` on bad input."""
    if not isinstance(row, dict):
        raise ValueError("pricing row must be an object")
    match = row.get("match")
    if not isinstance(match, str) or not match.strip():
        raise ValueError("'match' must be a non-empty string")
    match = match.strip().lower()

    def _price(field: str) -> float:
        value = row.get(field)
        try:
            price = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{field}' must be a number") from None
        if not math.isfinite(price) or price < 0:
            raise ValueError(f"'{field}' must be a finite non-negative number")
        return price

    input_price = _price("input")
    output_price = _price("output")

    display_name = row.get("display_name")
    if display_name is not None and not isinstance(display_name, str):
        raise ValueError("'display_name' must be a string")

    normalized: Dict[str, Any] = {
        "match": match,
        "input": input_price,
        "output": output_price,
    }
    if display_name:
        normalized["display_name"] = display_name
    return normalized


class PricingTable:
    """Thread-safe merged cloud-pricing table (bundled defaults + user rows).

    Mirrors :class:`UsageLedger`'s locking style. Bundled defaults ship as
    package data (`DEFAULT_PRICING_FILE`); a user can add, edit, or delete
    rows at runtime via the admin API/UI or by editing the JSON file at
    ``user_file`` directly — user rows override a builtin with the same
    ``match`` key, or extend the table with a new one. The merged view is
    rebuilt after every mutation, and the user file's mtime is checked on
    read so a direct on-disk edit is picked up without a server restart.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._user_file = base_path / USER_PRICING_FILENAME if base_path else None
        self._defaults: List[Dict[str, Any]] = _load_default_pricing_rows()
        self._user_rows: List[Dict[str, Any]] = []
        self._merged: List[Dict[str, Any]] = []
        self._mtime: Optional[float] = None
        with self._lock:
            self._reload_locked()

    @property
    def user_file(self) -> Optional[Path]:
        """Path to the user-editable overrides file, or ``None`` if unset."""
        return self._user_file

    def _current_mtime_locked(self) -> Optional[float]:
        if not self._user_file:
            return None
        try:
            return self._user_file.stat().st_mtime
        except OSError:
            return None

    def _read_user_rows_locked(self) -> List[Dict[str, Any]]:
        if not self._user_file or not self._user_file.exists():
            return []
        try:
            with open(self._user_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load user cloud pricing file %s: %s", self._user_file, e
            )
            return []
        raw_rows = data.get("rows") if isinstance(data, dict) else data
        if not isinstance(raw_rows, list):
            logger.warning(
                "User cloud pricing file %s has an unexpected shape; ignoring",
                self._user_file,
            )
            return []
        rows = []
        for row in raw_rows:
            try:
                rows.append(_validate_pricing_row(row))
            except ValueError as e:
                logger.warning(
                    "Skipping invalid pricing row in %s: %s", self._user_file, e
                )
        return rows

    def _reload_locked(self) -> None:
        """Recompute the merged view. Must be called while holding the lock."""
        self._user_rows = self._read_user_rows_locked()
        self._mtime = self._current_mtime_locked()
        merged: Dict[str, Dict[str, Any]] = {
            row["match"]: dict(row, source="builtin") for row in self._defaults
        }
        for row in self._user_rows:
            merged[row["match"]] = dict(row, source="user")
        self._merged = list(merged.values())

    def _ensure_fresh_locked(self) -> None:
        """Reload if the user file changed on disk since the last read."""
        if self._current_mtime_locked() != self._mtime:
            self._reload_locked()

    def load(self) -> List[Dict[str, Any]]:
        """Return the merged pricing rows (user rows win by ``match``)."""
        with self._lock:
            self._ensure_fresh_locked()
            return [dict(row) for row in self._merged]

    def rows(self, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """Return rows for the admin UI, tagged with source/override info.

        Args:
            include_builtin: When ``False``, only user-defined rows are
                returned (builtins that a user row overrides are omitted
                entirely, not just hidden).
        """
        with self._lock:
            self._ensure_fresh_locked()
            user_matches = {row["match"] for row in self._user_rows}
            default_matches = {row["match"] for row in self._defaults}
            result: List[Dict[str, Any]] = []
            if include_builtin:
                for row in self._defaults:
                    tagged = dict(row)
                    tagged["source"] = "builtin"
                    tagged["overridden"] = row["match"] in user_matches
                    result.append(tagged)
            for row in self._user_rows:
                tagged = dict(row)
                tagged["source"] = "user"
                tagged["overridden"] = row["match"] in default_matches
                result.append(tagged)
            return result

    def add_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a user pricing row. Validates and persists.

        Args:
            row: ``{"match", "input", "output", "display_name"?}``.

        Returns:
            The normalized, persisted row.

        Raises:
            ValueError: if ``row`` fails validation, or no base path is
                configured to persist to.
        """
        validated = _validate_pricing_row(row)
        if not self._user_file:
            raise ValueError(
                "No base path configured for the pricing table; cannot persist rows"
            )
        with self._lock:
            self._ensure_fresh_locked()
            rows = [r for r in self._user_rows if r["match"] != validated["match"]]
            rows.append(validated)
            self._write_user_rows_locked(rows)
            self._reload_locked()
        return validated

    def delete_row(self, match: str) -> bool:
        """Remove a user row, restoring any builtin default it overrode.

        Returns:
            ``True`` if a user row was removed, ``False`` if ``match`` only
            exists as a builtin (or not at all) — nothing to delete.
        """
        normalized = (match or "").strip().lower()
        if not self._user_file:
            return False
        with self._lock:
            self._ensure_fresh_locked()
            if not any(r["match"] == normalized for r in self._user_rows):
                return False
            rows = [r for r in self._user_rows if r["match"] != normalized]
            self._write_user_rows_locked(rows)
            self._reload_locked()
        return True

    def _write_user_rows_locked(self, rows: List[Dict[str, Any]]) -> None:
        """Atomically persist ``rows`` to the user file (tmp + replace)."""
        assert self._user_file is not None
        self._user_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self._user_file.with_name(
            f"{self._user_file.name}.{os.getpid()}.tmp"
        )
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(self._user_file)
        except OSError as e:
            logger.warning(
                "Failed to save user cloud pricing file %s: %s", self._user_file, e
            )
            temp_file.unlink(missing_ok=True)
            raise

    def lookup(self, model_id: str) -> Optional[Dict[str, float]]:
        """Find the pricing row whose key is the longest substring match.

        Longest-match-wins so a more specific key (``gpt-4o-mini``) is
        picked over a shorter one it happens to contain (``gpt-4o``). This
        preserves the exact semantics of the pre-refactor `_lookup_pricing`.
        """
        if not model_id:
            return None
        with self._lock:
            self._ensure_fresh_locked()
            rows = self._merged
        normalized = model_id.lower()
        best_row: Optional[Dict[str, Any]] = None
        for row in rows:
            key = row["match"]
            if key in normalized and (
                best_row is None or len(key) > len(best_row["match"])
            ):
                best_row = row
        if best_row is None:
            return None
        return {"input": best_row["input"], "output": best_row["output"]}


# Global singleton, mirroring omlx.server_metrics's pattern.
_pricing_table: Optional[PricingTable] = None


def get_pricing_table() -> PricingTable:
    """Get the global PricingTable singleton.

    Lazily initialized against `resolve_default_base_path()` so callers
    (e.g. admin routes) don't need the server startup path to have run
    `reset_pricing_table()` first.
    """
    global _pricing_table
    if _pricing_table is None:
        _pricing_table = PricingTable(resolve_default_base_path())
    return _pricing_table


def reset_pricing_table(base_path: Optional[Path] = None) -> None:
    """(Re)point the global pricing table singleton at ``base_path``.

    Called on server start (real base path) and by tests (a tmp_path, or
    ``None`` to isolate from any on-disk user overrides).
    """
    global _pricing_table
    _pricing_table = PricingTable(base_path)


def estimate_api_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int
) -> Optional[float]:
    """Estimate USD cost of serving this usage via a paid cloud API.

    Returns ``None`` (not ``0.0``) when ``model_id`` isn't in the pricing
    table, so "we don't know" is never confused with "free".
    """
    pricing = get_pricing_table().lookup(model_id)
    if pricing is None:
        return None
    cost = (
        prompt_tokens / 1_000_000 * pricing["input"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )
    return round(cost, 6)


def estimate_per_model_cost(per_model: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Sum estimated cost across a per-model breakdown dict.

    Returns ``None`` if none of the models are recognized, otherwise sums
    the recognized ones (unpriced models are silently excluded from the
    total rather than making the whole total unknown).
    """
    total = 0.0
    matched = False
    for model_id, counters in per_model.items():
        cost = estimate_api_cost(
            model_id,
            counters.get("prompt_tokens", 0),
            counters.get("completion_tokens", 0),
        )
        if cost is not None:
            total += cost
            matched = True
    return round(total, 6) if matched else None


class UsageLedger:
    """Thread-safe append-only ledger of closed server sessions.

    One record is appended per closed session (server restart or clean
    shutdown); the currently-open session is never written here — callers
    compute it live from :class:`~omlx.server_metrics.ServerMetrics`. The
    backing file is capped to the last ``max_sessions`` records.
    """

    def __init__(
        self,
        ledger_path: Optional[Path] = None,
        max_sessions: int = MAX_LEDGER_SESSIONS,
    ):
        self._lock = threading.Lock()
        self._ledger_path = ledger_path
        self._max_sessions = max_sessions

    def record_session_close(self, record: Dict[str, Any]) -> None:
        """Append a closed-session record to the ledger. Thread-safe.

        No-op if no ledger path is configured (e.g. tests, or a server
        run with no persistence directory).
        """
        if not self._ledger_path:
            return
        with self._lock:
            try:
                self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._ledger_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                self._enforce_cap()
            except OSError as e:
                logger.warning(
                    "Failed to append usage ledger record to %s: %s",
                    self._ledger_path,
                    e,
                )

    def _enforce_cap(self) -> None:
        """Trim the ledger file to the last ``max_sessions`` lines.

        Must be called while holding ``self._lock``.
        """
        assert self._ledger_path is not None
        try:
            lines = self._ledger_path.read_text().splitlines()
        except OSError:
            return
        if len(lines) <= self._max_sessions:
            return
        trimmed = lines[-self._max_sessions :]
        try:
            tmp_path = self._ledger_path.with_suffix(".jsonl.tmp")
            tmp_path.write_text("\n".join(trimmed) + "\n")
            tmp_path.replace(self._ledger_path)
        except OSError as e:
            logger.warning(
                "Failed to trim usage ledger %s: %s", self._ledger_path, e
            )

    def load_sessions(self) -> List[Dict[str, Any]]:
        """Return closed session records, newest first. Thread-safe."""
        if not self._ledger_path or not self._ledger_path.exists():
            return []
        with self._lock:
            try:
                lines = self._ledger_path.read_text().splitlines()
            except OSError as e:
                logger.warning(
                    "Failed to read usage ledger from %s: %s", self._ledger_path, e
                )
                return []
        sessions: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        sessions.reverse()
        return sessions


def close_current_session(metrics: "Any") -> None:
    """Close out ``metrics``'s current session into the global ledger.

    Takes a :class:`~omlx.server_metrics.ServerMetrics` instance (typed as
    ``Any`` to avoid a circular import) and appends its
    ``to_session_record()`` to :func:`get_usage_ledger`. A no-op if the
    session recorded zero requests, so restarts before any traffic don't
    pollute the ledger with empty rows.
    """
    record = metrics.to_session_record()
    if record.get("requests", 0) <= 0:
        return
    get_usage_ledger().record_session_close(record)


# Global singleton, mirroring omlx.server_metrics's pattern.
_usage_ledger: Optional[UsageLedger] = None


def get_usage_ledger() -> UsageLedger:
    """Get the global UsageLedger singleton."""
    global _usage_ledger
    if _usage_ledger is None:
        _usage_ledger = UsageLedger()
    return _usage_ledger


def reset_usage_ledger(ledger_path: Optional[Path] = None) -> None:
    """(Re)point the global ledger singleton at ``ledger_path``.

    Called on server start; does not itself close a session — call
    :func:`close_current_session` with the outgoing metrics first.
    """
    global _usage_ledger
    _usage_ledger = UsageLedger(ledger_path=ledger_path)
