# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for cluster v2 pairing (Module B).

No network, no SSH, no Module A code: enrollment/revocation drivers, SSH key
provider, caps, clocks, and HTTP transport are all fakes.  The loopback
happy path wires two real PairingManagers together through their public
joiner/coordinator APIs.
"""

import base64
import json
import stat
import threading
from types import SimpleNamespace

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
    EnrollmentDriveError,
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
    default_enrollment_driver,
    generate_pairing_code,
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


def _test_public_key(node_id: str) -> str:
    payload = base64.b64encode(f"key-{node_id}".encode()).decode()
    return f"ssh-ed25519 {payload}"


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
    enrollment_driver=None,
    revocation_driver=None,
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
        address_provider=lambda: ["127.0.0.1"],
        ssh_key_provider=lambda: _test_public_key(node_id),
        ssh_host_key_provider=lambda: _test_public_key(f"host-{node_id}"),
        enrollment_driver=enrollment_driver or driver,
        revocation_driver=revocation_driver or revocation,
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
    joiner = _manager(
        tmp_path,
        node_id="join-node",
        name="Joiner",
        clock=joiner_clock,
        driver_calls=driver_calls,
    )

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


def test_code_hash_authenticates_node_and_ssh_identities():
    code, node_id = "042517", "node-abc"
    user_key = _test_public_key(node_id)
    host_key = _test_public_key(f"host-{node_id}")
    salt = b"s" * 16
    digest = pairing_code_hash(code, node_id, user_key, host_key, salt)

    assert len(digest) == 64
    assert pairing_code_hash(code, "other", user_key, host_key, salt) != digest
    assert (
        pairing_code_hash(code, node_id, _test_public_key("other"), host_key, salt)
        != digest
    )
    assert (
        pairing_code_hash(code, node_id, user_key, _test_public_key("other"), salt)
        != digest
    )
    assert pairing_code_hash(code, node_id, user_key, host_key, b"x" * 16) != digest


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
    assert driver_calls[0]["ssh_public_key"] == _test_public_key("join-node")

    # Joiner polls, unwraps, persists: both sides hold the same cluster key.
    status = joiner.poll_join("coordinator.local:8080")
    assert status["state"] == "approved"
    joined = joiner.complete_join(status)
    assert joined["node_id"] == "coord-node"
    assert [call["node_id"] for call in driver_calls] == [
        "join-node",
        "coord-node",
    ]

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
    coordinator = _manager(
        tmp_path, node_id="coord-node", name="Coordinator", clock=clock
    )
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))

    wrong = "000000" if code != "000000" else "000001"
    for _attempt in range(1, MAX_CODE_ATTEMPTS):
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
    coordinator = _manager(
        tmp_path, node_id="coord-node", name="Coordinator", clock=clock
    )
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
    coordinator = _manager(
        tmp_path, node_id="coord-node", name="Coordinator", clock=clock
    )
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


# --- Unpair revocation ------------------------------------------------------------


def test_unpair_revokes_everything(tmp_path):
    coordinator, joiner, enrollment_store, _, revocation_calls = _loopback_pair(
        tmp_path
    )
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
            "peer_public_key": _test_public_key("join-node"),
            "addrs": ["127.0.0.1"],
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


def test_approval_reservation_blocks_deny_unpair_and_duplicate_approval(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocking_enrollment(_peer):
        entered.set()
        assert release.wait(5), "test did not release enrollment"
        return {"ok": True}

    coordinator = _manager(
        tmp_path,
        node_id="coord-node",
        name="Coordinator",
        enrollment_driver=blocking_enrollment,
    )
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))
    result = []
    errors = []

    def approve():
        try:
            result.append(coordinator.approve("join-node", code))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=approve)
    thread.start()
    assert entered.wait(5), "approval never reached enrollment"

    with pytest.raises(PairingStateError, match="already in progress"):
        coordinator.approve("join-node", code)
    with pytest.raises(PairingStateError, match="already in progress"):
        coordinator.deny("join-node")
    with pytest.raises(PairingStateError, match="already in progress"):
        coordinator.unpair("join-node")

    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert result[0]["state"] == "paired"
    assert coordinator._key_store.get("join-node") is not None


def test_persistence_failure_revokes_enrollment_and_releases_reservation(tmp_path):
    revocations = []
    coordinator = _manager(
        tmp_path,
        node_id="coord-node",
        name="Coordinator",
        revocation_calls=revocations,
    )
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))
    coordinator._key_store.set = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError("read-only pairing store")
    )

    with pytest.raises(PairingError, match="could not be persisted"):
        coordinator.approve("join-node", code)

    assert coordinator.paired_devices() == []
    assert coordinator._key_store.get("join-node") is None
    assert coordinator.pending_requests()[0]["approving"] is False
    assert revocations == [
        {
            "peer_public_key": _test_public_key("join-node"),
            "addrs": ["127.0.0.1"],
        }
    ]
    assert "approve_persistence_failed" in coordinator._audit.names()


def test_device_persistence_failure_rolls_back_cluster_key(tmp_path):
    revocations = []
    coordinator = _manager(
        tmp_path,
        node_id="coord-node",
        name="Coordinator",
        revocation_calls=revocations,
    )
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    code = joiner.start_join()["code"]
    coordinator.handle_join_request(joiner.build_join_request(code))
    coordinator._devices.put_paired = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError("read-only device registry")
    )

    with pytest.raises(PairingError, match="could not be persisted"):
        coordinator.approve("join-node", code)

    assert coordinator._key_store.get("join-node") is None
    assert coordinator.paired_devices() == []
    assert coordinator.pending_requests()[0]["approving"] is False
    assert len(revocations) == 1


def test_joiner_enrollment_failure_does_not_persist_coordinator(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    code = joiner.start_join()["code"]
    joiner.request_join("coordinator.local:8080")
    coordinator.approve("join-node", code)
    joiner._enrollment_driver = lambda _peer: (_ for _ in ()).throw(
        RuntimeError("host key mismatch")
    )

    with pytest.raises(EnrollmentDriveError, match="coordinator was not paired"):
        joiner.complete_join(joiner.poll_join("coordinator.local:8080"))

    assert joiner.paired_devices() == []
    assert joiner._key_store.get("coord-node") is None
    assert "join_enrollment_failed" in joiner._audit.names()


def test_joiner_persistence_failure_revokes_enrollment_and_keeps_retry_state(
    tmp_path,
):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    revocations = []
    joiner._revocation_driver = lambda material: revocations.append(material) or {}
    code = joiner.start_join()["code"]
    joiner.request_join("coordinator.local:8080")
    coordinator.approve("join-node", code)
    status = joiner.poll_join("coordinator.local:8080")
    joiner._key_store.set = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError("read-only pairing store")
    )

    with pytest.raises(PairingError, match="coordinator was not paired"):
        joiner.complete_join(status)

    assert joiner.paired_devices() == []
    assert joiner._key_store.get("coord-node") is None
    assert joiner._local_code["completing"] is False
    assert revocations[-1] == {
        "peer_public_key": _test_public_key("coord-node"),
        "addrs": ["127.0.0.1"],
    }
    assert "join_persistence_failed" in joiner._audit.names()


def test_join_completion_rejects_an_expired_local_code(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    code = joiner.start_join()["code"]
    joiner.request_join("coordinator.local:8080")
    coordinator.approve("join-node", code)
    status = joiner.poll_join("coordinator.local:8080")
    joiner._clock.now += CODE_TTL_SECONDS + 1

    with pytest.raises(PairingExpiredError, match="expired"):
        joiner.complete_join(status)

    assert joiner._local_code is None
    assert joiner.paired_devices() == []


def test_approval_requires_coordinator_ssh_material(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    code = joiner.start_join()["code"]
    joiner.request_join("coordinator.local:8080")
    coordinator._address_provider = lambda: []

    with pytest.raises(EnrollmentDriveError, match="peer was not paired"):
        coordinator.approve("join-node", code)

    assert coordinator.paired_devices() == []
    assert [row["node_id"] for row in coordinator.pending_requests()] == ["join-node"]


def test_default_enrollment_requires_key_and_verified_address(monkeypatch):
    from omlx.cluster import ssh_keys

    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(fingerprint="SHA256:local"),
    )
    with pytest.raises(PairingRequestError, match="SSH public key"):
        default_enrollment_driver({"ssh_public_key": None, "addrs": ["127.0.0.1"]})
    with pytest.raises(EnrollmentDriveError, match="verified peer address"):
        default_enrollment_driver(
            {
                "ssh_public_key": _test_public_key("peer"),
                "ssh_host_public_key": _test_public_key("host-peer"),
                "addrs": [],
            }
        )


def test_default_enrollment_propagates_host_key_failure(monkeypatch):
    from omlx.cluster import ssh_keys

    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(fingerprint="SHA256:local"),
    )
    monkeypatch.setattr(ssh_keys, "install_authorized_key", lambda **_kwargs: True)
    monkeypatch.setattr(
        ssh_keys,
        "pin_enrolled_host_key",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("refusing changed SSH host key")
        ),
    )

    with pytest.raises(RuntimeError, match="changed SSH host key"):
        default_enrollment_driver(
            {
                "ssh_public_key": _test_public_key("peer"),
                "ssh_host_public_key": _test_public_key("host-peer"),
                "addrs": ["127.0.0.1"],
            }
        )


def test_default_enrollment_formats_ipv6_known_host_target(monkeypatch):
    from omlx.cluster import ssh_keys

    targets: list[str] = []
    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(fingerprint="SHA256:local"),
    )
    monkeypatch.setattr(ssh_keys, "install_authorized_key", lambda **_kwargs: True)
    monkeypatch.setattr(
        ssh_keys,
        "pin_enrolled_host_key",
        lambda *, hostname, public_key: targets.append(hostname) or True,
    )

    default_enrollment_driver(
        {
            "ssh_public_key": _test_public_key("peer"),
            "ssh_host_public_key": _test_public_key("host-peer"),
            "addrs": ["::1"],
        }
    )

    assert targets == ["[::1]"]


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
    with pytest.raises(PairingRequestError, match="code_salt"):
        coordinator.handle_join_request(good | {"code_salt": "not-base64"})
    with pytest.raises(PairingRequestError, match="friendly_name"):
        coordinator.handle_join_request(good | {"friendly_name": ""})
    with pytest.raises(PairingRequestError, match="caps"):
        coordinator.handle_join_request(good | {"caps": ["not", "a", "dict"]})
    with pytest.raises(PairingRequestError, match="SSH public key"):
        coordinator.handle_join_request(
            good
            | {
                "ssh_public_key": (
                    _test_public_key("join-node") + "\n" + _test_public_key("attacker")
                )
            }
        )
    with pytest.raises(PairingRequestError, match="http_port"):
        coordinator.handle_join_request(good | {"http_port": 70000})
    with pytest.raises(PairingRequestError, match="6 digits"):
        coordinator.approve("join-node", "12345")


def test_pending_requests_are_memory_only_and_capped(tmp_path):
    clock = _Clock()
    coordinator = _manager(
        tmp_path, node_id="coord-node", name="Coordinator", clock=clock
    )
    for index in range(5):
        joiner = _manager(tmp_path, node_id=f"join-{index}", name=f"J{index}")
        code = joiner.start_join()["code"]
        coordinator.handle_join_request(joiner.build_join_request(code))
    assert len(coordinator.pending_requests()) == 5
    # Nothing pending was persisted to devices.json.
    store = JsonDeviceStore(tmp_path / "coord-node")
    assert store.list_paired() == []


def test_pending_request_is_idempotent_but_cannot_be_overwritten(tmp_path):
    coordinator = _manager(tmp_path, node_id="coord-node", name="Coordinator")
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    joiner.start_join()
    request = joiner.build_join_request()

    first = coordinator.handle_join_request(request)
    repeated = coordinator.handle_join_request(dict(request))
    assert repeated == first

    with pytest.raises(PairingStateError, match="different join request"):
        coordinator.handle_join_request(
            request | {"ssh_host_public_key": _test_public_key("attacker-host")}
        )


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


def test_endpoints_full_flow(tmp_path, monkeypatch):
    client, manager, _ = _client(tmp_path)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    paired_marks: list[str] = []
    monkeypatch.setattr(
        "omlx.cluster.discovery.get_discovery_service",
        lambda: SimpleNamespace(mark_paired=paired_marks.append),
    )
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
    assert paired_marks == ["join-node"]

    status = client.get("/api/cluster/pair/status/join-node").json()
    assert status["state"] == "approved"
    assert status["coordinator"]["node_id"] == "coord-node"
    joined = joiner.complete_join(status, code)
    assert joined["node_id"] == "coord-node"

    response = client.delete("/api/cluster/devices/join-node")
    assert response.status_code == 200
    assert response.json()["unpaired"] is True
    assert client.get("/api/cluster/pair/status/join-node").json()["state"] == "unknown"


def test_public_pairing_endpoints_are_rate_limited(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path)
    joiner = _manager(tmp_path, node_id="join-node", name="Joiner")
    payload = joiner.build_join_request(joiner.start_join()["code"])
    monkeypatch.setattr(
        pairing_routes.pair_request_rate_limiter,
        "allow",
        lambda _client: False,
    )
    assert client.post("/api/cluster/pair/request", json=payload).status_code == 429

    monkeypatch.setattr(
        pairing_routes.pair_status_rate_limiter,
        "allow",
        lambda _client: False,
    )
    assert client.get("/api/cluster/pair/status/join-node").status_code == 429


def test_pair_request_binds_enrollment_to_http_source(tmp_path):
    captured: list[dict] = []

    class _CaptureManager:
        @staticmethod
        def handle_join_request(payload):
            captured.append(payload)
            return {"state": "awaiting_approval"}

    pairing_routes.set_pairing_manager_getter(lambda: _CaptureManager())
    app = FastAPI()
    app.include_router(pairing_routes.pair_router)
    client = TestClient(app, client=("198.51.100.23", 50000))
    payload = {
        "node_id": "join-node",
        "friendly_name": "Joiner",
        "caps": {},
        "code_hash": "a" * 64,
        "code_salt": base64.b64encode(b"0" * 16).decode(),
        "http_port": 8000,
        "addrs": ["203.0.113.99"],
        "ssh_public_key": _test_public_key("join-node"),
        "ssh_host_public_key": _test_public_key("host-join-node"),
    }

    response = client.post("/api/cluster/pair/request", json=payload)

    assert response.status_code == 202
    assert captured[0]["addrs"] == ["198.51.100.23"]


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

    assert (
        client.post(
            "/api/cluster/pair/request", json=good | {"code_hash": "x"}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/cluster/pair/request", json=good | {"extra": 1}).status_code
        == 422
    )
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
    shown = joiner.start_join()
    joiner.request_join("coord:8000")
    coordinator.approve("join-node", shown["code"])

    status = coordinator.join_status("join-node")
    assert status["state"] == "approved"
    assert status["coordinator"]["caps"] == {"chip": "M4 Max", "ram_gb": 128}
    assert status["coordinator"]["ssh_public_key"] == _test_public_key("coord-node")
    assert status["coordinator"]["ssh_host_public_key"] == _test_public_key(
        "host-coord-node"
    )

    record = joiner.complete_join(status)
    assert record["caps"] == {"chip": "M4 Max", "ram_gb": 128}


def test_join_status_rejects_substituted_coordinator_identity(tmp_path):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    shown = joiner.start_join()
    joiner.request_join("coord:8000")
    coordinator.approve("join-node", shown["code"])
    status = coordinator.join_status("join-node")
    status["coordinator"] = dict(status["coordinator"])
    status["coordinator"]["ssh_host_public_key"] = _test_public_key("attacker-host")

    with pytest.raises(PairingCodeError, match="identity was altered"):
        joiner.complete_join(status)

    assert joiner.paired_devices() == []
