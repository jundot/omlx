# SPDX-License-Identifier: Apache-2.0
"""Serialized, admin-owned joiner session for the dashboard."""

from __future__ import annotations

import hashlib
import secrets
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .pairing import PairingError, PairingRequestError, PairingStateError

if TYPE_CHECKING:
    from .pairing import PairingManager


class PairingSession:
    def __init__(self, manager: PairingManager):
        self.manager = manager
        self.lock = threading.RLock()
        self.mutation_lock = threading.RLock()
        self.attempt: dict[str, Any] | None = None
        self.polling = False

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            empty = {
                "state": "idle",
                "code": None,
                "expires_at": None,
                "coordinator_addr": None,
                "error": None,
                "seconds_remaining": 0,
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
        with self.lock:
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
            raise PairingRequestError(
                "The coordinator could not accept the join request."
            ) from exc
        with self.lock:
            if self.attempt is attempt:
                self.manager._record_audit(
                    "join_requested",
                    node_id=self.manager.node_id,
                    detail={"coordinator": normalized},
                )
            return self.snapshot()

    def poll(self) -> dict[str, Any]:
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
                else:
                    self.attempt["error"] = None
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
            if attempt is None:
                return self.snapshot()
            token = attempt.get("cancel_token")
            attempt["state"] = "cancelling"
        # Wait for an outbound join POST before withdrawing it. Poll responses
        # cannot complete this attempt once its state is cancelling.
        try:
            if token:
                self.manager._http_post(
                    f"http://{attempt['coordinator_addr']}/api/cluster/pair/request/cancel",
                    {"node_id": self.manager.node_id, "token": token},
                    10.0,
                )
        except Exception as exc:
            with self.lock:
                attempt.update(
                    state="error",
                    code=None,
                    error="Could not cancel the join. Retry when the coordinator is reachable.",
                )
                self.manager._local_code = None
            raise PairingRequestError(attempt["error"]) from exc
        with self.lock:
            if self.attempt is attempt:
                self.manager._record_audit(
                    "join_cancelled", node_id=self.manager.node_id
                )
                self.attempt = None
                self.manager._local_code = None
            return self.snapshot()
