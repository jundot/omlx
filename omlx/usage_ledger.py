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
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sessions retained in the JSONL ledger file. Oldest records are dropped
# once the file grows past this so a long-lived install doesn't accumulate
# an unbounded file across years of restarts.
MAX_LEDGER_SESSIONS = 1000

# Cloud API pricing in USD per 1M tokens, keyed by a lowercase substring
# matched against the served model_id (see `_lookup_pricing`). Figures are
# list prices for common hosted models as a rough basis for comparison —
# not a live pricing feed, and intentionally approximate.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


def _lookup_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """Find the pricing row whose key is the longest substring match.

    Longest-match-wins so a more specific key (``gpt-4o-mini``) is picked
    over a shorter one it happens to contain (``gpt-4o``).
    """
    if not model_id:
        return None
    normalized = model_id.lower()
    best_key: Optional[str] = None
    for key in MODEL_PRICING:
        if key in normalized and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICING[best_key] if best_key else None


def estimate_api_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int
) -> Optional[float]:
    """Estimate USD cost of serving this usage via a paid cloud API.

    Returns ``None`` (not ``0.0``) when ``model_id`` isn't in the built-in
    pricing table, so "we don't know" is never confused with "free".
    """
    pricing = _lookup_pricing(model_id)
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
