# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for cluster v2 pairing (Module B).

No network, no SSH, no Module A code: enrollment/revocation drivers, SSH key
provider, caps, clocks, and HTTP transport are all fakes.  The loopback
happy path wires two real PairingManagers together through their public
joiner/coordinator APIs.
"""

import hashlib
import json
import stat

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import pairing_routes
from omlx.cluster.pairing import (
    CODE_TTL_SECONDS,
    LOCKOUT_SECONDS,
    MAX_CODE_ATTEMPTS,
    PBKDF2_ITERATIONS,
    DeviceRegistryBridge,
    JsonDeviceStore,
    PairingAuditLog,
    PairingCodeError,
    PairingError,
    PairingExpiredError,
    PairingKeyStore,
    PairingLockoutError,
    PairingManager,
    PairingRequestError,
    PairingStateError,
    generate_pairing_code,
    normalize_coordinator_addr,
    pairing_code_hash,
    unwrap_cluster_key,
    wrap_cluster_key,
)


class _Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Audit:
    def __init__(self):
        self.events = []

    def __call__(self, event, *, node_id="", detail=None):
        self.events.append((event, node_id, detail or {}))

    def names(self):
        return [event for event, _, _ in self.events]


class _FakeEnrollmentStore:
    """Duck-type of the legacy ClusterEnrollmentStore bits pairing uses."""

    def __init__(self):
        self.removed = []

    def remove_node(self, node_id):
        self.removed.append(node_id)
        return True


class _FakeModuleARegistry:
    """Mimics Module A DeviceRegistry method names (upsert/get/remove/list)."""

    def __init__(self):
        self.records = {}

    def upsert(self, record):
        self.records[record["node_id"]] = dict(record)

    def get(self, node_id):
        return self.records.get(node_id)

    def remove(self, node_id):
        return self.records.pop(node_id, None) is not None

    def list(self):
        return list(self.records.values())


def _manager(
    tmp_path,
    *,
    node_id,
    name,
    clock=None,
    audit=None,
    registry=None,
    enrollment_store=None,
    driver_calls=None,
    revocation_calls=None,
):
    """A PairingManager with every side effect faked or sandboxed."""

    def driver(peer):
        if driver_calls is not None:
            driver_calls.append(peer)
        return {"ok": True}

    def revocation(material):
        if revocation_calls is not None:
            revocation_calls.append(material)
        return {"authorized_key_removed": True, "errors": []}

    return PairingManager(
        registry,
        enrollment_store,
        base_path=tmp_path / node_id,
        identity={
            "node_id": node_id,
            "friendly_name": name,
            "created_at": 1.0,
            "schema_version": 1,
        },
        caps_provider=lambda: {"chip": "M4 Max", "ram_gb": 128},
        ssh_key_provider=lambda: f"ssh-ed25519 AAAA-{node_id}",
        enrollment_driver=driver,
        revocation_driver=revocation,
        clock=clock or _Clock(),
        audit=audit if audit is not None else _Audit(),
    )


def _loopback_pair(tmp_path):
    """Coordinator + joiner managers joined by an in-process transport."""

    coordinator_clock, joiner_clock = _Clock(), _Clock()
    enrollment_store = _FakeEnrollmentStore()
    driver_calls, revocation_calls = [], []
    coordinator = _manager(
        tmp_path,
        node_id="coord-node",
        name="Coordinator",
        clock=coordinator_clock,
        enrollment_store=enrollment_store,
        driver_calls=driver_calls,
        revocation_calls=revocation_calls,
    )
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner", clock=joiner_clock)

    def http_post(url, payload, timeout):
        assert url.endswith("/api/cluster/pair/request")
        return coordinator.handle_join_request(payload)

    def http_get(url, timeout):
        node_id = url.rsplit("/", 1)[1]
        return coordinator.join_status(node_id)

    joiner._http_post = http_post
    joiner._http_get = http_get
    return coordinator, joiner, enrollment_store, driver_calls, revocation_calls


# --- Code + wrap crypto -------------------------------------------------------


def test_pairing_code_is_six_digits_with_leading_zeros():
    for _ in range(200):
        code = generate_pairing_code()
        assert code.isdigit() and len(code) == 6


def test_code_hash_matches_spec_blake2s():
    code, node_id = "042517", "node-abc"
    assert pairing_code_hash(code, node_id) == hashlib.blake2s(
        (code + node_id).encode("utf-8")
    ).hexdigest()
    # Bound to the node: a different node_id cannot reuse the hash.
    assert pairing_code_hash(code, "other") != pairing_code_hash(code, node_id)


def test_cluster_key_wrap_round_trip_and_wrong_code_fails_closed():
    key = b"\x5a" * 32
    package = wrap_cluster_key(key, "123456")
    assert package["kdf"] == "PBKDF2-HMAC-SHA256"
    assert package["iterations"] == PBKDF2_ITERATIONS
    assert unwrap_cluster_key(package, "123456") == key
    with pytest.raises(PairingCodeError):
        unwrap_cluster_key(package, "654321")


def test_cluster_key_wrap_detects_tampering():
    key = b"\x00" * 32
    package = wrap_cluster_key(key, "123456")
    tampered = dict(package)
    tampered["ciphertext"] = package["salt"]  # same-length valid b64, wrong bytes
    with pytest.raises(PairingCodeError):
        unwrap_cluster_key(tampered, "123456")


def test_wrap_rejects_oversized_iteration_counts():
    package = wrap_cluster_key(b"\x01" * 32, "123456")
    package["iterations"] = 10**9
    with pytest.raises(PairingRequestError):
        unwrap_cluster_key(package, "123456")


# --- Loopback happy path -------------------------------------------------------


def test_two_manager_loopback_happy_path(tmp_path):
    coordinator, joiner, enrollment_store, driver_calls, _ = _loopback_pair(tmp_path)

    shown = joiner.start_join()
    code = shown["code"]
    assert shown["expires_at"] - joiner._clock() == CODE_TTL_SECONDS

    result = joiner.request_join("coordinator.local:8080")
    assert result["state"] == "awaiting_approval"

    # Pending request is visible in the coordinator's devices view.
    view = coordinator.devices_view()
    assert [p["node_id"] for p in view["pending"]] == ["join-node"]
    assert view["pending"][0]["state"] == "awaiting_approval"

    # The code never crossed the wire, and the coordinator's snapshot of the
    # pending request does not echo even the hash back to the joiner.
    assert "code" not in json.dumps(result["response"])
    assert "code_hash" not in result["response"]

    approved = coordinator.approve("join-node", code)
    assert approved["state"] == "paired"
    assert driver_calls and driver_calls[0]["node_id"] == "join-node"
    assert driver_calls[0]["ssh_public_key"] == "ssh-ed25519 AAAA-join-node"

    # Joiner polls, unwraps, persists: both sides hold the same cluster key.
    status = joiner.poll_join("coordinator.local:8080")
    assert status["state"] == "approved"
    joined = joiner.complete_join(status)
    assert joined["node_id"] == "coord-node"

    coord_key = coordinator._key_store.get("join-node")["cluster_key"]
    joiner_key = joiner._key_store.get("coord-node")["cluster_key"]
    assert coord_key == joiner_key

    assert [d["node_id"] for d in coordinator.paired_devices()] == ["join-node"]
    assert [d["node_id"] for d in joiner.paired_devices()] == ["coord-node"]

    events = coordinator._audit.names()
    assert "join_request_received" in events
    assert "approve_success" in events
    assert "join_completed" in joiner._audit.names()


def test_request_join_without_start_join_fails(tmp_path):
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    with pytest.raises(PairingStateError, match="start_join"):
        joiner.build_join_request()


# --- Wrong-code lockout ---------------------------------------------------------


def test_wrong_code_lockout_then_code_dies(tmp_path):
    clock = _Clock()
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator", clock=clock)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))

    wrong = "000000" if code != "000000" else "000001"
    for attempt in range(1, MAX_CODE_ATTEMPTS):
        with pytest.raises(PairingCodeError, match="attempts left"):
            coordinator.approve("join-node", wrong)
    with pytest.raises(PairingLockoutError, match="locked until"):
        coordinator.approve("join-node", wrong)

    # Even the CORRECT code is refused during the lockout window.
    with pytest.raises(PairingLockoutError):
        coordinator.approve("join-node", code)
    assert coordinator._key_store.get("join-node") is None

    # Lockout and code TTL are both 10 minutes from request creation, so by
    # the time the lockout is served the code itself has expired — a locked
    # request can never become pairable again without a fresh request.
    clock.now += LOCKOUT_SECONDS + 1
    with pytest.raises(PairingExpiredError):
        coordinator.approve("join-node", code)

    events = coordinator._audit.names()
    assert events.count("approve_wrong_code") == MAX_CODE_ATTEMPTS - 1
    assert "approve_lockout" in events
    assert "approve_locked_out" in events


def test_attempts_reset_after_lockout_within_code_validity(tmp_path):
    """If a lockout ends while the code is still valid, attempts reset."""

    clock = _Clock()
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator", clock=clock)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))

    wrong = "000000" if code != "000000" else "000001"
    for _ in range(MAX_CODE_ATTEMPTS):
        with pytest.raises(PairingError):
            coordinator.approve("join-node", wrong)

    # Simulate a short lockout that ended well inside the code's validity.
    coordinator._pending["join-node"].locked_until = clock.now - 1
    approved = coordinator.approve("join-node", code)
    assert approved["state"] == "paired"
    assert coordinator._pending == {}


# --- Expiry ---------------------------------------------------------------------


def test_expired_request_cannot_be_approved(tmp_path):
    clock = _Clock()
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator", clock=clock)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))

    clock.now += CODE_TTL_SECONDS + 1
    with pytest.raises(PairingExpiredError):
        coordinator.approve("join-node", code)
    # Pruned: a second attempt reports "no pending request", not expiry.
    with pytest.raises(PairingStateError, match="no pending"):
        coordinator.approve("join-node", code)
    assert "join_request_expired" in coordinator._audit.names()


def test_joiner_side_expired_code_refused(tmp_path):
    clock = _Clock()
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner", clock=clock)
    joiner.start_join()
    clock.now += CODE_TTL_SECONDS + 1
    with pytest.raises(PairingExpiredError):
        joiner.build_join_request()


# --- Deny -----------------------------------------------------------------------


def test_deny_removes_pending_and_is_audited(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.start_join()
    joiner.request_join("coordinator.local:8080")

    assert coordinator.deny("join-node") is True
    assert coordinator.deny("join-node") is False
    assert coordinator.join_status("join-node")["state"] == "denied"
    assert coordinator.devices_view()["pending"] == []
    with pytest.raises(PairingStateError, match="denied"):
        coordinator.approve("join-node", "123456")
    assert "join_request_denied" in coordinator._audit.names()


# --- Joiner-side UI session (begin/poll/cancel) -------------------------------


def test_normalize_coordinator_addr():
    assert normalize_coordinator_addr("10.0.0.1") == "10.0.0.1:8000"
    assert normalize_coordinator_addr("10.0.0.1:9000") == "10.0.0.1:9000"
    assert normalize_coordinator_addr("http://10.0.0.1:9000/") == "10.0.0.1:9000"
    assert normalize_coordinator_addr("https://studio.local") == "studio.local:8000"
    # Bracketed IPv6 strips brackets; a bare v6 literal keeps the default port.
    assert normalize_coordinator_addr("[fe80::1]:9000") == "fe80::1:9000"
    assert normalize_coordinator_addr("fe80::1") == "fe80::1:8000"
    for bad in ("", "http://", "10.0.0.1:notaport", "10.0.0.1:70000", "[fe80::1"):
        with pytest.raises(PairingRequestError):
            normalize_coordinator_addr(bad)


def test_begin_join_success_returns_code_and_remembers_coordinator(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)

    snapshot = joiner.begin_join("coordinator.local")  # default port 8000
    assert snapshot["state"] == "awaiting_approval"
    assert snapshot["coordinator_addr"] == "coordinator.local:8000"
    assert snapshot["code"].isdigit() and len(snapshot["code"]) == 6
    assert snapshot["expires_at"] - joiner._clock() == CODE_TTL_SECONDS
    assert [p["node_id"] for p in coordinator.pending_requests()] == ["join-node"]
    assert "join_requested" in joiner._audit.names()

    state = joiner.local_join_state()
    assert state["state"] == "awaiting_approval"
    assert state["code"] == snapshot["code"]
    assert state["coordinator_addr"] == "coordinator.local:8000"
    assert 0 < state["seconds_remaining"] <= CODE_TTL_SECONDS
    assert state["error"] is None
    # The plaintext code only ever leaves the manager through this snapshot —
    # never the audit trail.
    assert snapshot["code"] not in json.dumps(joiner._audit.events)


def test_begin_join_transport_failure_clears_state(tmp_path):
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")

    def boom(url, payload, timeout):
        raise OSError("connection refused")

    joiner._http_post = boom
    with pytest.raises(PairingRequestError, match="unreachable or refused"):
        joiner.begin_join("10.9.9.9:8000")
    assert joiner.local_join_state()["state"] == "idle"
    assert joiner._local_code is None
    assert "join_request_failed" in joiner._audit.names()

    # A later attempt against a reachable coordinator works normally.
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator")
    joiner._http_post = (
        lambda url, payload, timeout: coordinator.handle_join_request(payload)
    )
    snapshot = joiner.begin_join("10.0.0.5")
    assert snapshot["state"] == "awaiting_approval"
    assert [p["node_id"] for p in coordinator.pending_requests()] == ["join-node"]


def test_begin_join_refuses_a_second_join_while_awaiting(tmp_path):
    _, joiner, *_ = _loopback_pair(tmp_path)
    joiner.begin_join("coordinator.local:8080")
    with pytest.raises(PairingStateError, match="already awaiting"):
        joiner.begin_join("coordinator.local:8080")


def test_local_join_state_idle_without_join(tmp_path):
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    state = joiner.local_join_state()
    assert state["state"] == "idle"
    assert state["code"] is None
    assert state["coordinator_addr"] is None
    # Polling with no join in progress never touches the transport.
    joiner._http_get = lambda url, timeout: (_ for _ in ()).throw(AssertionError)
    assert joiner.poll_join_once()["state"] == "idle"


def test_local_join_state_expired_code_becomes_error(tmp_path):
    _, joiner, *_ = _loopback_pair(tmp_path)
    joiner.begin_join("coordinator.local:8080")
    joiner._clock.now += CODE_TTL_SECONDS + 1

    state = joiner.local_join_state()
    assert state["state"] == "error"
    assert state["code"] is None  # an expired code is never shown again
    assert "expired" in state["error"]
    # The address survives so the UI can offer "start again" against it.
    assert state["coordinator_addr"] == "coordinator.local:8080"


def test_poll_join_once_approved_completes_and_persists(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    shown = joiner.begin_join("coordinator.local:8080")
    coordinator.approve("join-node", shown["code"])

    snapshot = joiner.poll_join_once()
    assert snapshot["state"] == "approved"
    assert snapshot["code"] is None  # the pairing is done — the code is gone
    assert snapshot["coordinator_addr"] == "coordinator.local:8080"
    assert snapshot["coordinator_name"] == "Coordinator"
    assert joiner._local_code is None
    assert "join_completed" in joiner._audit.names()

    # The coordinator landed in the device store on disk, not just memory.
    assert [d["node_id"] for d in joiner.paired_devices()] == ["coord-node"]
    reloaded = JsonDeviceStore(tmp_path / "join-node")
    assert reloaded.get("coord-node")["state"] == "paired"

    # Approved is reported exactly once; the next poll is back to idle.
    assert joiner.poll_join_once()["state"] == "idle"


def test_poll_join_once_denied_is_terminal_until_rejoin(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.begin_join("coordinator.local:8080")
    coordinator.deny("join-node")

    assert joiner.poll_join_once()["state"] == "denied"
    # Terminal: no further transport calls, state sticks until cancel/rejoin.
    joiner._http_get = lambda url, timeout: (_ for _ in ()).throw(AssertionError)
    assert joiner.poll_join_once()["state"] == "denied"

    assert joiner.cancel_join() == {"state": "idle"}
    fresh = joiner.begin_join("coordinator.local:8080")
    coordinator.approve("join-node", fresh["code"])
    joiner._http_get = (
        lambda url, timeout: coordinator.join_status(url.rsplit("/", 1)[1])
    )
    assert joiner.poll_join_once()["state"] == "approved"


def test_poll_join_once_transient_error_keeps_awaiting(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.begin_join("coordinator.local:8080")
    original_get = joiner._http_get

    def flaky(url, timeout):
        raise OSError("timed out")

    joiner._http_get = flaky
    snapshot = joiner.poll_join_once()
    assert snapshot["state"] == "awaiting_approval"
    assert "timed out" in snapshot["error"]
    assert snapshot["code"] is not None

    # The next successful poll clears the recorded error.
    joiner._http_get = original_get
    snapshot = joiner.poll_join_once()
    assert snapshot["state"] == "awaiting_approval"
    assert snapshot["error"] is None


def test_cancel_join_is_idempotent_and_rejoinable(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)

    # Nothing in progress: still idle, nothing audited.
    assert joiner.cancel_join() == {"state": "idle"}
    assert "join_cancelled" not in joiner._audit.names()

    joiner.begin_join("coordinator.local:8080")
    assert joiner.cancel_join() == {"state": "idle"}
    assert joiner.cancel_join() == {"state": "idle"}
    assert joiner._audit.names().count("join_cancelled") == 1
    assert joiner.local_join_state()["state"] == "idle"
    assert joiner._local_code is None

    # Re-join after cancel starts a fresh attempt against the coordinator.
    snapshot = joiner.begin_join("coordinator.local:8080")
    assert snapshot["state"] == "awaiting_approval"
    assert [p["node_id"] for p in coordinator.pending_requests()] == ["join-node"]


# --- Unpair revocation ------------------------------------------------------------


def test_unpair_revokes_everything(tmp_path):
    coordinator, joiner, enrollment_store, _, revocation_calls = _loopback_pair(tmp_path)
    code = joiner.start_join()["code"]
    joiner.request_join("coordinator.local:8080")
    coordinator.approve("join-node", code)
    joiner.complete_join(joiner.poll_join("coordinator.local:8080"))

    result = coordinator.unpair("join-node")
    assert result["unpaired"] is True
    assert result["removed_device"] is True
    assert result["removed_key"] is True
    assert result["removed_enrollment"] is True
    assert coordinator.paired_devices() == []
    assert coordinator._key_store.get("join-node") is None
    assert enrollment_store.removed == ["join-node"]
    assert revocation_calls == [
        {
            "peer_public_key": "ssh-ed25519 AAAA-join-node",
            "addrs": [],
        }
    ]
    assert coordinator.join_status("join-node")["state"] == "unknown"
    assert "device_unpaired" in coordinator._audit.names()

    # The joiner's own copy is revoked independently.
    joiner_result = joiner.unpair("coord-node")
    assert joiner_result["removed_key"] is True
    assert joiner.paired_devices() == []


def test_unpair_unknown_device_fails_closed(tmp_path):
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator")
    with pytest.raises(PairingStateError, match="unknown device"):
        coordinator.unpair("ghost-node")


def test_unpair_also_cancels_a_pending_request(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    joiner.start_join()
    joiner.request_join("coordinator.local:8080")
    result = coordinator.unpair("join-node")
    assert result["was_pending"] is True
    assert coordinator.devices_view()["pending"] == []


# --- Fail-closed enrollment driving -----------------------------------------------


def test_enrollment_failure_does_not_pair(tmp_path):
    def boom(peer):
        raise RuntimeError("refusing changed SSH host key")

    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator")
    coordinator._enrollment_driver = boom
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))

    from omlx.cluster.pairing import EnrollmentDriveError

    with pytest.raises(EnrollmentDriveError, match="not paired"):
        coordinator.approve("join-node", code)
    assert coordinator.paired_devices() == []
    assert coordinator._key_store.get("join-node") is None
    # Pending survives so the operator can fix SSH and retry with the code.
    assert [p["node_id"] for p in coordinator.pending_requests()] == ["join-node"]
    assert "approve_enrollment_failed" in coordinator._audit.names()


# --- Input validation -------------------------------------------------------------


def test_join_request_validation(tmp_path):
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator")
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    joiner.start_join()
    good = joiner.build_join_request()

    with pytest.raises(PairingRequestError, match="own cluster"):
        coordinator.handle_join_request(good | {"node_id": "coord-node"})
    with pytest.raises(PairingRequestError, match="code_hash"):
        coordinator.handle_join_request(good | {"code_hash": "zz"})
    with pytest.raises(PairingRequestError, match="friendly_name"):
        coordinator.handle_join_request(good | {"friendly_name": ""})
    with pytest.raises(PairingRequestError, match="caps"):
        coordinator.handle_join_request(good | {"caps": ["not", "a", "dict"]})
    with pytest.raises(PairingRequestError, match="http_port"):
        coordinator.handle_join_request(good | {"http_port": 70000})
    with pytest.raises(PairingRequestError, match="6 digits"):
        coordinator.approve("join-node", "12345")


def test_pending_requests_are_memory_only_and_capped(tmp_path):
    clock = _Clock()
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator", clock=clock)
    for index in range(5):
        joiner = _manager(tmp_path, node_id=f"join-{index}", name=f"J{index}")
        code = joiner.start_join()["code"]
        coordinator.handle_join_request(joiner.build_join_request(code))
    assert len(coordinator.pending_requests()) == 5
    # Nothing pending was persisted to devices.json.
    store = JsonDeviceStore(tmp_path / "coord-node")
    assert store.list_paired() == []


# --- Persistence ------------------------------------------------------------------


def test_key_store_persists_with_private_permissions(tmp_path):
    store = PairingKeyStore(tmp_path)
    store.set("node-a", {"cluster_key": "ab" * 32, "paired_at": 1.0})
    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert mode == 0o600

    reloaded = PairingKeyStore(tmp_path)
    assert reloaded.get("node-a")["cluster_key"] == "ab" * 32
    assert reloaded.remove("node-a") is not None
    assert PairingKeyStore(tmp_path).get("node-a") is None


def test_key_store_fails_closed_on_corruption(tmp_path):
    store = PairingKeyStore(tmp_path)
    store.set("node-a", {"cluster_key": "ab" * 32})
    store.path.write_text("{not json")
    reloaded = PairingKeyStore(tmp_path)
    assert reloaded.get("node-a") is None
    assert reloaded.load_error is not None


def test_device_store_round_trip_and_permissions(tmp_path):
    store = JsonDeviceStore(tmp_path)
    store.put_paired(
        {
            "node_id": "n1",
            "friendly_name": "Peer",
            "caps": {},
            "paired_at": 1.0,
            "last_addrs": ["10.0.0.2"],
        }
    )
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    payload = json.loads(store.path.read_text())
    assert payload["schema_version"] == 1
    reloaded = JsonDeviceStore(tmp_path)
    assert reloaded.get("n1")["state"] == "paired"
    assert reloaded.remove("n1") is True
    assert reloaded.remove("n1") is False


def test_fallback_identity_persists_node_id(tmp_path):
    first = pairing_load(tmp_path)
    second = pairing_load(tmp_path)
    assert first["node_id"] == second["node_id"]
    path = tmp_path / "cluster" / "identity.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["schema_version"] == 1


def pairing_load(base_path):
    from omlx.cluster.pairing import load_node_identity

    return load_node_identity(base_path)


def test_audit_log_appends_json_lines(tmp_path):
    log = PairingAuditLog(tmp_path / "cluster" / "pairing-audit.jsonl")
    log.record("approve_success", node_id="n1")
    log.record("device_unpaired", node_id="n1", detail={"removed_key": True})
    lines = log.path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["event"] == "device_unpaired"
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o600


# --- Module A bridge ---------------------------------------------------------------


def test_module_a_registry_is_used_via_feature_detection(tmp_path):
    registry = _FakeModuleARegistry()
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    coordinator_with_a = _manager(
        tmp_path, node_id="coord-b", name="CoordB", registry=registry
    )
    joiner2 = _manager(tmp_path, node_id="join-b", name="JoinB")
    code = joiner2.start_join()["code"]
    coordinator_with_a.handle_join_request(joiner2.build_join_request(code))
    coordinator_with_a.approve("join-b", code)

    assert registry.records["join-b"]["state"] == "paired"
    assert [d["node_id"] for d in coordinator_with_a.paired_devices()] == ["join-b"]
    coordinator_with_a.unpair("join-b")
    assert registry.records == {}


def test_registry_bridge_reports_api_drift_loudly():
    bridge = DeviceRegistryBridge(object())
    with pytest.raises(PairingStateError, match="DeviceRegistry exposes none of"):
        bridge.put_paired({"node_id": "n"})


def test_registry_bridge_against_real_module_a_registry(tmp_path):
    """Integration lock: the bridge drives Module A's real DeviceRegistry.

    mark_paired (not merge) must persist the peer as paired — merge would
    leave a newly approved device memory-only.
    """

    from omlx.cluster.registry import DeviceRegistry

    registry = DeviceRegistry(tmp_path / "devices.json")
    bridge = DeviceRegistryBridge(registry)
    bridge.put_paired(
        {
            "node_id": "peer-real",
            "friendly_name": "studio",
            "caps": {"chip": "M3"},
            "paired_at": 42.0,
            "last_addrs": ["192.168.5.2"],
            "state": "paired",
        }
    )

    assert registry.is_paired("peer-real")
    stored = bridge.get("peer-real")
    assert stored["friendly_name"] == "studio"
    assert stored["paired_at"] == 42.0
    assert [d["node_id"] for d in bridge.list_paired()] == ["peer-real"]
    # Persisted to disk, not memory-only.
    reloaded = DeviceRegistry(tmp_path / "devices.json")
    assert reloaded.is_paired("peer-real")
    assert bridge.remove("peer-real") is True
    assert not registry.is_paired("peer-real")


# --- HTTP endpoints -----------------------------------------------------------------


def _client(tmp_path):
    clock = _Clock()
    manager = _manager(tmp_path, node_id="coord-node", name="Coordinator", clock=clock)
    pairing_routes.set_pairing_manager_getter(lambda: manager)
    app = FastAPI()
    app.include_router(pairing_routes.pair_router)
    app.include_router(pairing_routes.pair_admin_router)
    return TestClient(app), manager, clock


@pytest.fixture(autouse=True)
def _restore_manager_getter():
    yield
    pairing_routes.set_pairing_manager_getter(None)


def test_endpoints_full_flow(tmp_path):
    client, manager, _ = _client(tmp_path)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    payload = joiner.build_join_request(code)

    response = client.post("/api/cluster/pair/request", json=payload)
    assert response.status_code == 202
    assert response.json()["state"] == "awaiting_approval"

    # Wrong code → 403, then lockout → 423.
    wrong = "000000" if code != "000000" else "000001"
    for _ in range(MAX_CODE_ATTEMPTS):
        response = client.post(
            "/api/cluster/pair/approve", json={"node_id": "join-node", "code": wrong}
        )
    assert response.status_code == 423
    response = client.post(
        "/api/cluster/pair/approve", json={"node_id": "join-node", "code": code}
    )
    assert response.status_code == 423

    # Re-request after lockout is over; approve succeeds.
    manager._pending["join-node"].locked_until = manager._clock() - 1
    response = client.post(
        "/api/cluster/pair/approve", json={"node_id": "join-node", "code": code}
    )
    assert response.status_code == 200
    assert response.json()["state"] == "paired"

    status = client.get("/api/cluster/pair/status/join-node").json()
    assert status["state"] == "approved"
    assert status["coordinator"]["node_id"] == "coord-node"
    joined = joiner.complete_join(status, code)
    assert joined["node_id"] == "coord-node"

    response = client.delete("/api/cluster/devices/join-node")
    assert response.status_code == 200
    assert response.json()["unpaired"] is True
    assert client.get("/api/cluster/pair/status/join-node").json()["state"] == "unknown"


def test_endpoint_error_mapping(tmp_path):
    client, manager, clock = _client(tmp_path)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]

    # Approve with no pending request → 404.
    response = client.post(
        "/api/cluster/pair/approve", json={"node_id": "ghost", "code": "123456"}
    )
    assert response.status_code == 404

    client.post("/api/cluster/pair/request", json=joiner.build_join_request(code))
    wrong = "000000" if code != "000000" else "000001"
    response = client.post(
        "/api/cluster/pair/approve", json={"node_id": "join-node", "code": wrong}
    )
    assert response.status_code == 403

    # Expiry → 410.
    clock.now += CODE_TTL_SECONDS + 1
    response = client.post(
        "/api/cluster/pair/approve", json={"node_id": "join-node", "code": code}
    )
    assert response.status_code == 410

    # Deny unknown → 404; delete unknown → 404.
    assert (
        client.post("/api/cluster/pair/deny", json={"node_id": "ghost"}).status_code
        == 404
    )
    assert client.delete("/api/cluster/devices/ghost").status_code == 404


def test_endpoint_deny_flow(tmp_path):
    client, manager, _ = _client(tmp_path)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    joiner.start_join()
    client.post("/api/cluster/pair/request", json=joiner.build_join_request())

    response = client.post("/api/cluster/pair/deny", json={"node_id": "join-node"})
    assert response.status_code == 200
    assert client.get("/api/cluster/pair/status/join-node").json()["state"] == "denied"


def test_request_validation_rejects_bad_payloads(tmp_path):
    client, _, _ = _client(tmp_path)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    joiner.start_join()
    good = joiner.build_join_request()

    assert client.post("/api/cluster/pair/request", json=good | {"code_hash": "x"}).status_code == 422
    assert client.post("/api/cluster/pair/request", json=good | {"extra": 1}).status_code == 422
    assert (
        client.post(
            "/api/cluster/pair/approve", json={"node_id": "n", "code": "12345"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/cluster/pair/request", json=good | {"node_id": "coord-node"}
        ).status_code
        == 400
    )


# --- Joiner-side endpoints (/pair/join) ------------------------------------------


def _joiner_client(tmp_path):
    """Routers mounted over a JOINER manager looping back to a coordinator."""

    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator")

    def http_post(url, payload, timeout):
        assert url.endswith("/api/cluster/pair/request")
        return coordinator.handle_join_request(payload)

    def http_get(url, timeout):
        return coordinator.join_status(url.rsplit("/", 1)[1])

    joiner._http_post = http_post
    joiner._http_get = http_get
    pairing_routes.set_pairing_manager_getter(lambda: joiner)
    app = FastAPI()
    app.include_router(pairing_routes.pair_admin_router)
    return TestClient(app), joiner, coordinator


def test_join_endpoints_full_joiner_flow(tmp_path):
    client, joiner, coordinator = _joiner_client(tmp_path)

    # Idle until a join begins; the cancel endpoint is idempotent.
    assert client.get("/api/cluster/pair/join").json()["state"] == "idle"
    assert client.post("/api/cluster/pair/join/cancel").json() == {"state": "idle"}

    response = client.post("/api/cluster/pair/join", json={"coordinator_addr": "10.0.0.5"})
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["state"] == "awaiting_approval"
    assert snapshot["coordinator_addr"] == "10.0.0.5:8000"  # default port
    code = snapshot["code"]
    assert code.isdigit() and len(code) == 6

    # The 1 Hz UI poll keeps reporting awaiting_approval (and drives completion).
    waiting = client.get("/api/cluster/pair/join").json()
    assert waiting["state"] == "awaiting_approval"
    assert waiting["code"] == code

    coordinator.approve("join-node", code)
    approved = client.get("/api/cluster/pair/join").json()
    assert approved["state"] == "approved"
    assert approved["code"] is None
    assert approved["coordinator_name"] == "Coordinator"
    assert [d["node_id"] for d in joiner.paired_devices()] == ["coord-node"]
    # Approval is reported exactly once, then the snapshot is idle again.
    assert client.get("/api/cluster/pair/join").json()["state"] == "idle"


def test_join_endpoint_reports_denial(tmp_path):
    client, _, coordinator = _joiner_client(tmp_path)
    client.post("/api/cluster/pair/join", json={"coordinator_addr": "10.0.0.5"})

    coordinator.deny("join-node")
    assert client.get("/api/cluster/pair/join").json()["state"] == "denied"

    # Cancel clears back to idle so the UI panel closes.
    assert client.post("/api/cluster/pair/join/cancel").json() == {"state": "idle"}
    assert client.get("/api/cluster/pair/join").json()["state"] == "idle"


def test_join_endpoint_transport_failure_maps_to_400(tmp_path):
    client, joiner, _ = _joiner_client(tmp_path)

    def boom(url, payload, timeout):
        raise OSError("connection refused")

    joiner._http_post = boom
    response = client.post("/api/cluster/pair/join", json={"coordinator_addr": "10.9.9.9"})
    assert response.status_code == 400
    assert "unreachable or refused" in response.json()["detail"]
    # The failed attempt left no local join state behind.
    assert client.get("/api/cluster/pair/join").json()["state"] == "idle"


def test_join_endpoints_validate_payloads(tmp_path):
    client, _, _ = _joiner_client(tmp_path)

    assert client.post("/api/cluster/pair/join", json={}).status_code == 422
    assert (
        client.post("/api/cluster/pair/join", json={"coordinator_addr": ""}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/cluster/pair/join",
            json={"coordinator_addr": "10.0.0.1", "extra": 1},
        ).status_code
        == 422
    )
    # Well-formed body, malformed address → PairingRequestError → 400.
    assert (
        client.post("/api/cluster/pair/join", json={"coordinator_addr": "http://"}).status_code
        == 400
    )


def test_unconfigured_manager_returns_503(tmp_path):
    pairing_routes.set_pairing_manager_getter(
        lambda: (_ for _ in ()).throw(RuntimeError("cluster pairing is not configured"))
    )
    app = FastAPI()
    app.include_router(pairing_routes.pair_router)
    app.include_router(pairing_routes.pair_admin_router)
    client = TestClient(app)
    assert client.get("/api/cluster/pair/status/x").status_code == 503
    assert client.delete("/api/cluster/devices/x").status_code == 503
    assert (
        client.post(
            "/api/cluster/pair/join", json={"coordinator_addr": "10.0.0.5"}
        ).status_code
        == 503
    )
    assert client.get("/api/cluster/pair/join").status_code == 503
    assert client.post("/api/cluster/pair/join/cancel").status_code == 503


# --- Legacy non-regression ----------------------------------------------------------


def test_legacy_pairing_endpoints_still_registered():
    from omlx.cluster import routes

    paths = {
        (route.path, tuple(sorted(route.methods)))
        for route in routes.router.routes
        for methods in [getattr(route, "methods", set())]
        if methods
    }
    expected = {
        "/admin/api/cluster/pairing-token",
        "/admin/api/cluster/verify-pairing-token",
        "/admin/api/cluster/ssh-key",
        "/admin/api/cluster/ssh-key/generate",
        "/admin/api/cluster/ssh-key/exchange-token",
        "/admin/api/cluster/ssh-key/exchange",
        "/admin/api/cluster/ssh-key/store-keychain",
        "/admin/api/cluster/join-keys",
        "/admin/api/cluster/join-status",
    }
    present = {path for path, _ in paths}
    assert expected <= present


def test_legacy_pairing_token_flow_still_works():
    from omlx.cluster.discovery import generate_pairing_token, verify_pairing_token

    secret = "x" * 32
    token = generate_pairing_token(shared_secret=secret)
    assert verify_pairing_token(token, shared_secret=secret) is True
    assert verify_pairing_token(token, shared_secret="y" * 32) is False


def test_join_status_carries_coordinator_caps_and_key(tmp_path):
    """The poll path is the only one the UI drives: complete_join persists
    whatever join_status returns, so the coordinator block must carry the
    same caps/ssh key that approve() returns synchronously."""

    coordinator, joiner, _, _, _ = _loopback_pair(tmp_path)
    joiner.begin_join("coord:8000")
    coordinator.approve("join-node", joiner._local_code["code"])

    status = coordinator.join_status("join-node")
    assert status["state"] == "approved"
    assert status["coordinator"]["caps"] == {"chip": "M4 Max", "ram_gb": 128}
    assert status["coordinator"]["ssh_public_key"] == "ssh-ed25519 AAAA-coord-node"

    record = joiner.complete_join(status)
    assert record["caps"] == {"chip": "M4 Max", "ram_gb": 128}
