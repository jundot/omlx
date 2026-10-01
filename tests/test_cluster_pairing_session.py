# SPDX-License-Identifier: Apache-2.0
"""Join recovery across process restarts and unreachable coordinators."""

import errno
import io
import json
import socket
import stat
from urllib.error import HTTPError, URLError

import pytest

from omlx.cluster.pairing import PairingRequestError
from tests.test_cluster_pairing import _loopback_pair, _manager


def _restart(tmp_path, old):
    manager = _manager(
        tmp_path, node_id=old.node_id, name=old.friendly_name, clock=old._clock
    )
    manager._http_post = old._http_post
    manager._http_get = old._http_get
    manager._enrollment_driver = old._enrollment_driver
    return manager


def test_restart_preserves_code_and_cancel_proof(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    shown = joiner.ui_session.begin("coordinator:8000")
    token = joiner.ui_session.attempt["cancel_token"]
    restored = _restart(tmp_path, joiner)
    assert restored.ui_session.snapshot() == shown
    assert restored.ui_session.attempt["cancel_token"] == token
    assert "cancel_token" not in restored.ui_session.snapshot()
    assert stat.S_IMODE(restored.ui_session.path.stat().st_mode) == 0o600
    assert restored.ui_session.cancel()["state"] == "idle"
    assert not coordinator.pending_requests()
    restored.ui_session.begin("coordinator:8000")
    assert restored.ui_session.attempt["cancel_token"] != token


def test_restart_can_complete_approval_with_original_code(tmp_path):
    coordinator, joiner, _, enrollments, _ = _loopback_pair(tmp_path)
    shown = joiner.ui_session.begin("coordinator:8000")
    restored = _restart(tmp_path, joiner)
    coordinator.approve(joiner.node_id, shown["code"])
    assert restored.ui_session.poll()["state"] == "approved"
    assert len(enrollments) == 2
    saved = json.loads(restored.ui_session.path.read_text())
    assert "code" not in saved["attempt"]
    assert "cancel_token" not in saved["attempt"]
    assert saved["withdrawals"] == []


def test_cancel_offline_survives_restart_and_allows_other_peer(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    other = _manager(tmp_path, node_id="other", name="Other")
    original_post = joiner._http_post
    joiner.ui_session.begin("coordinator:8000")
    token = joiner.ui_session.attempt["cancel_token"]

    def offline(*args):
        raise ConnectionRefusedError("coordinator offline")

    joiner._http_post = offline
    assert joiner.ui_session.cancel() == {
        "state": "idle",
        "code": None,
        "expires_at": None,
        "coordinator_addr": None,
        "error": None,
        "seconds_remaining": 0,
        "cleanup_pending": True,
    }
    restored = _restart(tmp_path, joiner)
    assert restored._local_code is None
    assert restored.ui_session.withdrawals[0]["cancel_token"] == token
    sent = []

    def route(url, payload, timeout):
        sent.append(url)
        if "other:8000" in url:
            return other.handle_join_request(payload)
        return original_post(url, payload, timeout)

    restored._http_post = route
    restored._http_get = lambda *_: other.join_status(restored.node_id)
    shown = restored.ui_session.begin("other:8000")
    assert sent == ["http://other:8000/api/cluster/pair/request"]
    assert shown["cleanup_pending"] is True
    assert restored.ui_session.poll()["state"] == "awaiting_approval"
    assert not coordinator.pending_requests()
    assert other.pending_requests()
    assert restored.ui_session.snapshot()["cleanup_pending"] is False


def test_cleanup_retries_are_bounded_and_same_peer_does_not_mint_new_token(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original = joiner._http_post
    joiner.ui_session.begin("coordinator:8000")
    calls = []

    def offline(url, payload, timeout):
        calls.append((url, timeout))
        raise ConnectionRefusedError("offline")

    joiner._http_post = offline
    joiner.ui_session.cancel()
    for _ in range(5):
        assert joiner.ui_session.poll()["state"] == "idle"
    assert len(calls) == 1
    assert calls[0][1] == 2.0
    joiner._clock.now += 5
    joiner.ui_session.poll()
    assert len(calls) == 2
    with pytest.raises(PairingRequestError, match="not confirmed cleanup"):
        joiner.ui_session.begin("coordinator:8000")
    assert all(url.endswith("/request/cancel") for url, _ in calls)
    joiner._http_post = original
    assert joiner.ui_session.begin("coordinator:8000")["state"] == "awaiting_approval"
    assert len(coordinator.pending_requests()) == 1


def test_rejected_rejoin_does_not_claim_ownership_of_old_request(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.start_join()  # An older process did not persist its UI session.
    coordinator.handle_join_request(joiner.build_join_request())
    old_request = coordinator._pending[joiner.node_id]
    with pytest.raises(PairingRequestError, match="deny it on that Mac"):
        joiner.ui_session.begin("coordinator:8000")
    assert "cancel_token" not in joiner.ui_session.attempt
    assert joiner.ui_session.cancel()["state"] == "idle"
    assert coordinator._pending[joiner.node_id] is old_request
    assert coordinator.deny(joiner.node_id)
    assert joiner.ui_session.begin("coordinator:8000")["state"] == "awaiting_approval"


@pytest.mark.parametrize("status", [400, 403, 404, 409, 429, 500])
def test_http_rejection_and_uncertain_server_error_keep_different_proofs(
    tmp_path, status
):
    _, joiner, *_ = _loopback_pair(tmp_path)

    def fail(url, *_):
        raise HTTPError(url, status, "request failed", {}, None)

    joiner._http_post = fail
    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin("coordinator:8000")
    assert ("cancel_token" in joiner.ui_session.attempt) is (status == 500)
    assert joiner.ui_session.cancel()["state"] == "idle"
    assert joiner.ui_session.snapshot()["cleanup_pending"] is (status == 500)


def test_lost_response_proof_survives_restart(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original = joiner._http_post

    def lost(url, payload, timeout):
        original(url, payload, timeout)
        raise TimeoutError("response lost")

    joiner._http_post = lost
    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin("coordinator:8000")
    restored = _restart(tmp_path, joiner)
    restored._http_post = original
    restored.ui_session.begin("coordinator:8000")
    assert len(coordinator.pending_requests()) == 1


def test_coordinator_restart_does_not_leave_join_waiting_for_missing_request(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.ui_session.begin("coordinator:8000")
    coordinator._pending.clear()
    result = joiner.ui_session.poll()
    assert result["state"] == "error"
    assert "no longer has this join" in result["error"]
    assert joiner.ui_session.begin("coordinator:8000")["state"] == "awaiting_approval"


def test_expired_restored_join_does_not_restore_usable_code(tmp_path):
    _, joiner, *_ = _loopback_pair(tmp_path)
    shown = joiner.ui_session.begin("coordinator:8000")
    joiner._clock.now = shown["expires_at"] + 1
    restored = _restart(tmp_path, joiner)
    assert restored.ui_session.snapshot()["state"] == "error"
    assert restored._local_code is None
    assert _restart(tmp_path, restored).ui_session.snapshot()["code"] is None


def test_join_storage_failure_does_not_send_request(tmp_path, monkeypatch):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(joiner.ui_session, "_save", fail)
    with pytest.raises(OSError, match="disk full"):
        joiner.ui_session.begin("coordinator:8000")
    assert not coordinator.pending_requests()
    assert joiner.ui_session.snapshot()["state"] == "idle"
    assert joiner._local_code is None


def test_cancel_storage_failure_preserves_original_attempt(tmp_path, monkeypatch):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.ui_session.begin("coordinator:8000")
    attempt = joiner.ui_session.attempt

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(joiner.ui_session, "_save", fail)
    with pytest.raises(OSError, match="disk full"):
        joiner.ui_session.cancel()
    assert joiner.ui_session.attempt is attempt
    assert joiner._local_code is not None
    assert coordinator.pending_requests()


def test_superseded_proof_cannot_remove_new_peer_request(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.ui_session.begin("coordinator:8000")
    old_attempt = dict(joiner.ui_session.attempt)
    joiner.ui_session.cancel()
    joiner.ui_session.begin("coordinator:8000")
    new_request = coordinator._pending[joiner.node_id]
    # Simulate an old saved proof meeting a peer that has a newer request.
    joiner.ui_session.attempt = old_attempt
    assert joiner.ui_session.cancel()["state"] == "idle"
    assert not joiner.ui_session.withdrawals
    assert coordinator._pending[joiner.node_id] is new_request


@pytest.mark.parametrize(
    "reason",
    [
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        socket.gaierror(socket.EAI_NONAME, "unknown host"),
    ],
)
def test_failed_initial_connection_does_not_require_remote_cleanup(tmp_path, reason):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original = joiner._http_post

    def fail(*args):
        raise URLError(reason)

    joiner._http_post = fail
    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin("coordinator:8000")
    assert not coordinator.pending_requests()
    assert "cancel_token" not in joiner.ui_session.attempt
    restored = _restart(tmp_path, joiner)
    calls = []

    def retry(url, payload, timeout):
        calls.append(url)
        return original(url, payload, timeout)

    restored._http_post = retry
    assert restored.ui_session.begin("coordinator:8000")["state"] == "awaiting_approval"
    assert calls == ["http://coordinator:8000/api/cluster/pair/request"]


@pytest.mark.parametrize(
    "reason", [TimeoutError("timeout"), ConnectionResetError("reset")]
)
def test_ambiguous_transport_failures_keep_cancellation_proof(tmp_path, reason):
    _, joiner, *_ = _loopback_pair(tmp_path)

    def fail(*args):
        raise URLError(reason)

    joiner._http_post = fail
    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin("coordinator:8000")
    assert joiner.ui_session.attempt.get("cancel_token")


def test_redirect_after_delivered_request_keeps_cancellation_proof(tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from omlx.cluster.pairing import _default_http_post

    coordinator, joiner, *_ = _loopback_pair(tmp_path)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            coordinator.handle_join_request(payload)
            self.send_response(302)
            self.send_header(
                "Location", "http://invalid.invalid/api/cluster/pair/request"
            )
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    joiner._http_post = _default_http_post
    try:
        with pytest.raises(PairingRequestError):
            joiner.ui_session.begin(f"127.0.0.1:{server.server_port}")
        assert coordinator.pending_requests()
        assert joiner.ui_session.attempt is not None
        assert joiner.ui_session.attempt["cancel_token"]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    "status, body, terminal",
    [
        (404, b'{"detail": "no pending join request"}', True),
        (404, b'{"detail": "Not Found"}', False),
        (404, b"<html>Not Found</html>", False),
        (404, b"", False),
        (404, b"null", False),
        (404, b'{"detail": ["no pending join request"]}', False),
        (404, b"\xff", False),
        (
            404,
            b'{"detail": "no pending join request", "padding": "' + b"x" * 4096 + b'"}',
            False,
        ),
        (403, b'{"detail": "invalid cancellation token"}', True),
        (409, b'{"detail": "no pending join request"}', False),
        (500, b'{"detail": "no pending join request"}', False),
    ],
    ids=[
        "missing-request",
        "missing-endpoint",
        "html",
        "empty",
        "null",
        "wrong-detail-type",
        "invalid-utf8",
        "oversized",
        "superseded",
        "conflict",
        "server-error",
    ],
)
def test_withdrawal_http_response_requires_explicit_missing_request(
    tmp_path, status, body, terminal
):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    original_post = joiner._http_post
    joiner.ui_session.begin("coordinator:8000")
    token = joiner.ui_session.attempt["cancel_token"]
    coordinator.deny(joiner.node_id)

    def http_post(url, payload, timeout):
        if url.endswith("/request/cancel"):
            raise HTTPError(url, status, "response", {}, io.BytesIO(body))
        return original_post(url, payload, timeout)

    joiner._http_post = http_post
    cancelled = joiner.ui_session.cancel()
    assert cancelled["state"] == "idle"
    assert cancelled["cleanup_pending"] is not terminal
    restored = _restart(tmp_path, joiner)
    assert restored.ui_session.snapshot()["cleanup_pending"] is not terminal
    if terminal:
        restored.ui_session.begin("coordinator:8000")
        assert restored.ui_session.attempt["cancel_token"] != token
    else:
        assert restored.ui_session.withdrawals[0]["cancel_token"] == token
        with pytest.raises(PairingRequestError, match="not confirmed cleanup"):
            restored.ui_session.begin("coordinator:8000")
        assert restored.ui_session.withdrawals[0]["cancel_token"] == token


def test_current_coordinator_missing_request_returns_success(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.ui_session.begin("coordinator:8000")
    coordinator.deny(joiner.node_id)
    assert joiner.ui_session.cancel()["cleanup_pending"] is False
    assert _restart(tmp_path, joiner).ui_session.snapshot()["cleanup_pending"] is False


def test_withdrawal_404_body_read_failure_retains_proof(tmp_path):
    _, joiner, *_ = _loopback_pair(tmp_path)
    joiner.ui_session.begin("coordinator:8000")
    token = joiner.ui_session.attempt["cancel_token"]

    class BrokenBody(io.BytesIO):
        def read(self, *args):
            raise OSError("response body unavailable")

    def http_post(url, payload, timeout):
        raise HTTPError(url, 404, "Not Found", {}, BrokenBody())

    joiner._http_post = http_post
    assert joiner.ui_session.cancel()["cleanup_pending"] is True
    restored = _restart(tmp_path, joiner)
    assert restored.ui_session.withdrawals[0]["cancel_token"] == token


def test_forget_cleanup_preserves_approved_pairing_and_survives_restart(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    shown = joiner.ui_session.begin("coordinator:8000")
    coordinator.approve(joiner.node_id, shown["code"])
    assert joiner.ui_session.poll()["state"] == "approved"
    session = joiner.ui_session
    session.withdrawals = [{"coordinator_addr": "old:8000", "cancel_token": "a" * 64}]
    session._save()
    attempt = dict(session.attempt)
    peers = joiner.paired_devices()

    def no_network(*args):
        pytest.fail("forget cleanup must not contact any peer")

    joiner._http_post = no_network
    joiner._http_get = no_network
    result = session.forget_cleanup()
    assert result["cleanup_pending"] is False
    assert result["state"] == "approved"
    assert session.attempt == attempt
    assert joiner.paired_devices() == peers
    assert _restart(tmp_path, joiner).ui_session.snapshot()["cleanup_pending"] is False
    assert session.forget_cleanup()["cleanup_pending"] is False


def test_forget_cleanup_save_failure_restores_pending_proof(tmp_path, monkeypatch):
    _, joiner, *_ = _loopback_pair(tmp_path)
    session = joiner.ui_session
    proof = {"coordinator_addr": "old:8000", "cancel_token": "a" * 64}
    session.withdrawals = [proof]
    session._save()
    saved = session.path.read_bytes()

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(session, "_save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        session.forget_cleanup()
    assert session.withdrawals == [proof]
    assert session.path.read_bytes() == saved


def test_forget_cleanup_endpoint_is_admin_only_and_reports_storage_failure(
    tmp_path, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import pairing_routes

    _, joiner, *_ = _loopback_pair(tmp_path)
    monkeypatch.setattr(pairing_routes, "_get_pairing_manager", lambda: joiner)
    path = "/api/cluster/pair/join/cleanup"
    assert path not in {route.path for route in pairing_routes.pair_router.routes}
    app = FastAPI()
    app.include_router(pairing_routes.pair_admin_router)
    client = TestClient(app)
    assert client.delete(path).json()["cleanup_pending"] is False
    joiner.ui_session.withdrawals = [
        {"coordinator_addr": "old:8000", "cancel_token": "a" * 64}
    ]

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(joiner.ui_session, "_save", fail_save)
    assert client.delete(path).status_code == 503
    assert joiner.ui_session.withdrawals


def test_forget_cleanup_serializes_with_inflight_retry(tmp_path):
    import threading

    _, joiner, *_ = _loopback_pair(tmp_path)
    session = joiner.ui_session
    session.withdrawals = [{"coordinator_addr": "old:8000", "cancel_token": "a" * 64}]
    session._save()
    entered = threading.Event()
    release = threading.Event()
    forgotten = threading.Event()
    errors = []

    def http_post(*args):
        entered.set()
        assert release.wait(5)
        raise OSError("peer offline")

    def forget():
        try:
            session.forget_cleanup()
        except Exception as exc:
            errors.append(exc)
        finally:
            forgotten.set()

    joiner._http_post = http_post
    retry = threading.Thread(target=session._retry_withdrawal)
    retry.start()
    assert entered.wait(5)
    worker = threading.Thread(target=forget)
    worker.start()
    try:
        assert not forgotten.wait(0.05)
    finally:
        release.set()
        retry.join(5)
        worker.join(5)
    assert not retry.is_alive()
    assert not worker.is_alive()
    assert not errors
    assert forgotten.is_set()
    assert session.snapshot()["cleanup_pending"] is False
    assert _restart(tmp_path, joiner).ui_session.snapshot()["cleanup_pending"] is False
