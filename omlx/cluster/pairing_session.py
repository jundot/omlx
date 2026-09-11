# SPDX-License-Identifier: Apache-2.0
"""Serialized, admin-owned joiner session for the dashboard."""

from __future__ import annotations

import hashlib
import secrets
import threading
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

from .pairing import (
    CODE_TTL_SECONDS,
    PairingCodeError,
    PairingError,
    PairingRequestError,
    PairingStateError,
    _atomic_write_json,
    _read_json_object,
)

if TYPE_CHECKING:
    from .pairing import PairingManager


class PairingSession:
    def __init__(self, manager: PairingManager):
        self.manager = manager
        self.lock = threading.RLock()
        self.mutation_lock = threading.RLock()
        self.attempt: dict[str, Any] | None = None
        self.polling = False
        self.withdrawals: list[dict[str, str]] = []
        self.cleanup_lock = threading.Lock()
        self.retry_after: dict[str, float] = {}
        self.path = manager.base_path / "cluster" / "join-session.json"
        if self.path.exists():
            saved = _read_json_object(self.path, 1, "join session")
            if saved.get("node_id") != manager.node_id:
                raise PairingStateError("join session belongs to a different node")
            self.attempt = saved.get("attempt")
            self.withdrawals = saved.get("withdrawals", [])
            if self.attempt and self.attempt.get("code"):
                manager._local_code = {
                    "code": self.attempt["code"],
                    "created_at": self.attempt["expires_at"] - CODE_TTL_SECONDS,
                    "expires_at": self.attempt["expires_at"],
                    "completing": False,
                }

    def _save(self) -> None:
        # Persist the original proof before sending a request or retiring it.
        # The shared atomic writer gives the code/token file mode 0600.
        _atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "node_id": self.manager.node_id,
                "attempt": self.attempt,
                "withdrawals": self.withdrawals,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            empty = {
                "state": "idle",
                "code": None,
                "expires_at": None,
                "coordinator_addr": None,
                "error": None,
                "seconds_remaining": 0,
                "cleanup_pending": bool(self.withdrawals),
            }
            if self.attempt is None:
                return empty
            result = empty | {
                key: value
                for key, value in self.attempt.items()
                if key != "cancel_token"
            }
            if result["state"] == "awaiting_approval":
                remaining = max(0, int(result["expires_at"] - self.manager._clock()))
                result["seconds_remaining"] = remaining
                if remaining == 0:
                    self.attempt = {
                        **self.attempt,
                        "state": "error",
                        "code": None,
                        "seconds_remaining": 0,
                        "error": "The pairing code expired. Start again.",
                    }
                    self.manager._local_code = None
                    self._save()
                    return self.snapshot()
            return result

    def begin(self, address: str) -> dict[str, Any]:
        with self.mutation_lock:
            with self.lock:
                cleanup = (
                    self.attempt is not None and self.snapshot()["state"] == "error"
                )
            if cleanup:
                self.cancel()
            return self._begin(address)

    def _begin(self, address: str) -> dict[str, Any]:
        raw = address.strip()
        try:
            parsed = urlsplit(raw if "://" in raw else "http://" + raw)
            valid = (
                parsed.scheme == "http"
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and parsed.path in {"", "/"}
                and not parsed.query
                and not parsed.fragment
            )
            port = 8000 if parsed.port is None else parsed.port
            if not valid or not 1 <= port <= 65535 or any(c.isspace() for c in raw):
                raise ValueError("invalid address")
        except ValueError as exc:
            raise PairingRequestError(
                "Use a coordinator hostname or IP and optional port."
            ) from exc
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        normalized = f"{host}:{port}"
        # Do not mint a replacement proof while the same peer may still have
        # the retired request. Other peers are independent of that cleanup.
        self._retry_withdrawal(address=normalized, force=True)
        with self.lock:
            if any(w["coordinator_addr"] == normalized for w in self.withdrawals):
                raise PairingRequestError(
                    "The previous join is cancelled on this Mac, but the other "
                    "Mac has not confirmed cleanup yet. Retry when it is reachable."
                )
            if self.snapshot()["state"] in {"awaiting_approval", "cancelling"}:
                raise PairingStateError(
                    "Cancel the existing join before starting another."
                )
            shown = self.manager.start_join()
            try:
                payload = self.manager.build_join_request()
            except Exception:
                self.manager._local_code = None
                self.attempt = None
                raise
            token = secrets.token_hex(32)
            payload["cancel_token_hash"] = hashlib.sha256(token.encode()).hexdigest()
            attempt = {
                "cancel_token": token,
                "state": "awaiting_approval",
                **shown,
                "coordinator_addr": normalized,
                "error": None,
            }
            self.attempt = attempt
            try:
                self._save()
            except Exception:
                self.attempt = None
                self.manager._local_code = None
                raise
        try:
            self.manager._http_post(
                f"http://{normalized}/api/cluster/pair/request", payload, 10.0
            )
        except Exception as exc:
            with self.lock:
                if self.attempt is not attempt:
                    return self.snapshot()
                self.manager._local_code = None
                attempt.update(
                    state="error",
                    code=None,
                    error="The join request failed. Retry or cancel.",
                )
                # A rejected request never owned the peer's pending record.
                # Keep proofs for timeouts/5xx: the response may have been lost.
                rejected = isinstance(exc, PairingError) or (
                    isinstance(exc, HTTPError) and 400 <= exc.code < 500
                )
                if rejected:
                    attempt.pop("cancel_token", None)
                    attempt["error"] = (
                        "The other Mac rejected this join. If an earlier join "
                        "is still pending there, deny it on that Mac and try again."
                    )
                self._save()
                message = attempt["error"]
            raise PairingRequestError(message) from exc
        with self.lock:
            if self.attempt is attempt:
                self.manager._record_audit(
                    "join_requested",
                    node_id=self.manager.node_id,
                    detail={"coordinator": normalized},
                )
            return self.snapshot()

    def poll(self) -> dict[str, Any]:
        self._retry_withdrawal()
        # Network I/O does not hold the session lock: cancellation can retire
        # this generation before a delayed approval arrives.
        with self.lock:
            current = self.snapshot()
            if current["state"] != "awaiting_approval" or self.polling:
                return current
            attempt = self.attempt
            self.polling = True
        try:
            status = self.manager.poll_join(current["coordinator_addr"])
            with self.lock:
                if (
                    self.attempt is not attempt
                    or self.snapshot()["state"] != "awaiting_approval"
                ):
                    return self.snapshot()
                if status.get("state") == "approved":
                    record = self.manager.complete_join(status)
                    self.attempt = {
                        "state": "approved",
                        "coordinator_addr": current["coordinator_addr"],
                        "coordinator_name": record.get("friendly_name", ""),
                    }
                elif status.get("state") == "denied":
                    self.manager._local_code = None
                    self.attempt = {
                        "state": "denied",
                        "coordinator_addr": current["coordinator_addr"],
                    }
                elif status.get("state") == "unknown":
                    self.manager._local_code = None
                    self.attempt = {
                        **attempt,
                        "state": "error",
                        "code": None,
                        "error": "The other Mac no longer has this join. Start again.",
                    }
                else:
                    self.attempt["error"] = None
                if self.attempt is not attempt:
                    self._save()
        except PairingError as exc:
            with self.lock:
                if self.attempt is attempt:
                    self.manager._local_code = None
                    self.attempt = {
                        **attempt,
                        "code": None,
                        "state": "error",
                        "error": str(exc),
                        "coordinator_addr": current["coordinator_addr"],
                    }
                    self._save()
        except Exception:
            with self.lock:
                if self.attempt is attempt:
                    self.attempt["error"] = (
                        "Coordinator status is temporarily unavailable."
                    )
        finally:
            with self.lock:
                self.polling = False
        return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        with self.mutation_lock:
            return self._cancel()

    def _cancel(self) -> dict[str, Any]:
        with self.lock:
            attempt = self.attempt
            if attempt is not None:
                old_withdrawals = self.withdrawals[:]
                local_code = self.manager._local_code
                if attempt.get("cancel_token"):
                    self.withdrawals.append(
                        {
                            "coordinator_addr": attempt["coordinator_addr"],
                            "cancel_token": attempt["cancel_token"],
                        }
                    )
                self.attempt = None
                self.manager._local_code = None
                try:
                    self._save()
                except Exception:
                    self.attempt = attempt
                    self.manager._local_code = local_code
                    self.withdrawals = old_withdrawals
                    raise
                self.manager._record_audit(
                    "join_cancelled", node_id=self.manager.node_id
                )
        # mutation_lock orders this after any outbound join POST. Retiring
        # locally first also prevents a delayed poll from completing the join.
        self._retry_withdrawal(
            address=attempt.get("coordinator_addr") if attempt else None,
            force=True,
        )
        return self.snapshot()

    def _retry_withdrawal(
        self, *, address: str | None = None, force: bool = False
    ) -> None:
        # The existing dashboard poll drives bounded cleanup; no worker or
        # network call is started merely by constructing a manager at boot.
        if not self.cleanup_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                now = self.manager._clock()
                pending = next(
                    (
                        w
                        for w in self.withdrawals
                        if (address is None or w["coordinator_addr"] == address)
                        and (
                            force
                            or now >= self.retry_after.get(w["coordinator_addr"], 0)
                        )
                    ),
                    None,
                )
                if pending is None:
                    return
                target = pending["coordinator_addr"]
                self.retry_after[target] = now + 5.0
            try:
                self.manager._http_post(
                    f"http://{target}/api/cluster/pair/request/cancel",
                    {"node_id": self.manager.node_id, "token": pending["cancel_token"]},
                    2.0,
                )
            except Exception as exc:
                # A different attempt owns the peer's record now. This proof
                # must never remove it, nor block local cancellation forever.
                if not isinstance(exc, PairingCodeError) and not (
                    isinstance(exc, HTTPError) and exc.code == 403
                ):
                    return
                self.manager._record_audit(
                    "join_cleanup_superseded",
                    node_id=self.manager.node_id,
                    detail={"coordinator": target},
                )
            with self.lock:
                self.withdrawals.remove(pending)
                try:
                    self._save()
                except Exception:
                    self.withdrawals.append(pending)
                    raise
        finally:
            self.cleanup_lock.release()
