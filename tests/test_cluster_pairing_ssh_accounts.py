# SPDX-License-Identifier: Apache-2.0
"""Account selection is authenticated before either peer installs SSH keys."""

from types import SimpleNamespace

import pytest

from omlx.cluster import pairing
from tests.test_cluster_pairing import _loopback_pair, _test_public_key


def _accounts(monkeypatch, user):
    monkeypatch.setattr(
        pairing.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name=user)
    )


def _request(tmp_path, monkeypatch):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    _accounts(monkeypatch, "joiner_user")
    shown = joiner.start_join(
        local_ssh_user="joiner_user", remote_ssh_user="coordinator_user"
    )
    payload = joiner.build_join_request()
    coordinator.handle_join_request(payload)
    return coordinator, joiner, shown["code"], payload


def test_selected_accounts_survive_symmetric_automatic_key_enrollment(
    tmp_path, monkeypatch
):
    coordinator, joiner, code, _ = _request(tmp_path, monkeypatch)
    from omlx.cluster import ssh_keys

    installed = []
    pinned = []
    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(fingerprint="SHA256:test"),
    )
    monkeypatch.setattr(
        ssh_keys,
        "install_authorized_key",
        lambda *, public_key: installed.append(public_key) or True,
    )
    monkeypatch.setattr(
        ssh_keys, "pin_enrolled_host_key", lambda **kw: pinned.append(kw) or True
    )
    coordinator._enrollment_driver = pairing.default_enrollment_driver
    joiner._enrollment_driver = pairing.default_enrollment_driver
    _accounts(monkeypatch, "coordinator_user")
    coordinator.approve(
        joiner.node_id,
        code,
        local_ssh_user="coordinator_user",
        remote_ssh_user="joiner_user",
    )
    assert installed == [_test_public_key("join-node")]
    _accounts(monkeypatch, "joiner_user")
    joiner.complete_join(coordinator.join_status(joiner.node_id))
    assert installed == [_test_public_key("join-node"), _test_public_key("coord-node")]
    assert len(pinned) == 2
    assert coordinator.paired_devices()[0]["ssh_user"] == "joiner_user"
    assert joiner.paired_devices()[0]["ssh_user"] == "coordinator_user"


@pytest.mark.parametrize(
    "local,remote",
    [("wrong_user", "peer_user"), ("joiner_user", None), ("joiner_user", "-option")],
)
def test_invalid_local_selection_stops_before_key_generation(
    tmp_path, monkeypatch, local, remote
):
    _, joiner, *_ = _loopback_pair(tmp_path)
    _accounts(monkeypatch, "joiner_user")
    joiner._ssh_key_provider = lambda: pytest.fail("must not access keys")
    with pytest.raises(pairing.PairingRequestError):
        joiner.start_join(local_ssh_user=local, remote_ssh_user=remote)
    assert joiner._local_code is None


@pytest.mark.parametrize(
    "field,value", [("ssh_user", "other_user"), ("expected_ssh_user", "other_user")]
)
def test_modified_account_cannot_authorize_key_installation(
    tmp_path, monkeypatch, field, value
):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    _accounts(monkeypatch, "joiner_user")
    code = joiner.start_join(
        local_ssh_user="joiner_user", remote_ssh_user="coordinator_user"
    )["code"]
    payload = joiner.build_join_request()
    payload[field] = value
    coordinator.handle_join_request(payload)
    coordinator._enrollment_driver = lambda peer: pytest.fail("must not install keys")
    _accounts(monkeypatch, payload["expected_ssh_user"])
    with pytest.raises(pairing.PairingCodeError):
        coordinator.approve(joiner.node_id, code)
    assert not coordinator.paired_devices()


def test_coordinator_account_mismatch_stops_before_enrollment(tmp_path, monkeypatch):
    coordinator, joiner, code, _ = _request(tmp_path, monkeypatch)
    _accounts(monkeypatch, "another_user")
    coordinator._enrollment_driver = lambda peer: pytest.fail("must not install keys")
    with pytest.raises(pairing.PairingRequestError):
        coordinator.approve(joiner.node_id, code)


def test_altered_approval_account_is_rejected_before_joiner_enrollment(
    tmp_path, monkeypatch
):
    coordinator, joiner, code, _ = _request(tmp_path, monkeypatch)
    _accounts(monkeypatch, "coordinator_user")
    coordinator.approve(joiner.node_id, code)
    status = coordinator.join_status(joiner.node_id)
    status["coordinator"]["ssh_user"] = "other_user"
    joiner._enrollment_driver = lambda peer: pytest.fail("must not install keys")
    _accounts(monkeypatch, "joiner_user")
    with pytest.raises(pairing.PairingCodeError):
        joiner.complete_join(status)
    assert not joiner.paired_devices()


def test_selected_accounts_survive_join_session_restart(tmp_path, monkeypatch):
    from tests.test_cluster_pairing_session import _restart

    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    _accounts(monkeypatch, "joiner_user")
    shown = joiner.ui_session.begin(
        "coordinator:8000",
        local_ssh_user="joiner_user",
        remote_ssh_user="coordinator_user",
    )
    restored = _restart(tmp_path, joiner)
    assert restored._local_code["remote_ssh_user"] == "coordinator_user"
    _accounts(monkeypatch, "coordinator_user")
    coordinator.approve(joiner.node_id, shown["code"])
    _accounts(monkeypatch, "joiner_user")
    assert restored.ui_session.poll()["state"] == "approved"
    assert restored.paired_devices()[0]["ssh_user"] == "coordinator_user"


def test_selected_account_is_persisted_by_real_registry(tmp_path):
    from omlx.cluster.registry import DeviceRegistry

    registry = DeviceRegistry(tmp_path / "devices.json")
    pairing.DeviceRegistryBridge(registry).put_paired(
        {
            "node_id": "peer",
            "friendly_name": "Peer",
            "last_addrs": ["192.0.2.1"],
            "ssh_user": "peer_user",
        }
    )
    restored = DeviceRegistry(tmp_path / "devices.json")
    assert restored.get("peer")["ssh_user"] == "peer_user"


def test_dotted_accounts_pair_and_persist(tmp_path, monkeypatch):
    coordinator, joiner, *_ = _loopback_pair(tmp_path)
    _accounts(monkeypatch, "joiner.user")
    shown = joiner.start_join(
        local_ssh_user="joiner.user", remote_ssh_user="coordinator.user"
    )
    coordinator.handle_join_request(joiner.build_join_request())
    _accounts(monkeypatch, "coordinator.user")
    coordinator.approve(
        joiner.node_id,
        shown["code"],
        local_ssh_user="coordinator.user",
        remote_ssh_user="joiner.user",
    )
    _accounts(monkeypatch, "joiner.user")
    joiner.complete_join(coordinator.join_status(joiner.node_id))
    assert coordinator.paired_devices()[0]["ssh_user"] == "joiner.user"
    assert joiner.paired_devices()[0]["ssh_user"] == "coordinator.user"
