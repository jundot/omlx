# SPDX-License-Identifier: Apache-2.0

import json
import stat

import pytest

from omlx.cluster.enrollment import (
    JOIN_SESSION_TTL_SECONDS,
    ClusterEnrollmentStore,
    EnrolledNode,
    EnrollmentError,
)


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _node(
    *, digest: str = "a" * 64, node_id: str = "cuda-worker-1-machine"
) -> EnrolledNode:
    return EnrolledNode(
        node_id=node_id,
        hostname="cuda-worker-1",
        ssh="omlxworker@10.42.0.21",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
        accelerator="cuda",
        platform="Linux-aarch64",
        python_executable="/opt/omlx-cluster-worker/venv/bin/python",
        source_digest=digest,
        ssh_host_fingerprint="SHA256:" + "A" * 43,
        joined_at=1001.0,
        last_seen_at=1001.0,
    )


def test_join_key_is_single_use_and_status_never_returns_the_secret(tmp_path):
    store = ClusterEnrollmentStore(tmp_path)
    raw_key, issued = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
    )

    raw_session, session = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )

    assert issued["status"] == "pending"
    assert session.source_digest == "a" * 64
    assert raw_key not in json.dumps(store.to_dict())
    assert raw_session not in json.dumps(store.to_dict())
    with pytest.raises(EnrollmentError, match="already been used"):
        store.claim(
            raw_key,
            node_id="cuda-worker-1-machine",
            hostname="cuda-worker-1",
            ssh_user="omlxworker",
            ssh_port=22,
            addresses=("10.42.0.21",),
        )


def test_default_join_key_survives_fresh_worker_prerequisite_install(tmp_path):
    clock = _Clock()
    store = ClusterEnrollmentStore(tmp_path, clock=clock)
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
    )

    clock.now += 15 * 60

    _, session = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )
    assert session.node_id == "cuda-worker-1-machine"


def test_expired_join_key_and_session_fail_closed(tmp_path):
    clock = _Clock()
    store = ClusterEnrollmentStore(tmp_path, clock=clock)
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
        ttl=30,
    )
    clock.now += 31
    with pytest.raises(EnrollmentError, match="invalid or expired"):
        store.claim(
            raw_key,
            node_id="cuda-worker-1-machine",
            hostname="cuda-worker-1",
            ssh_user="omlxworker",
            ssh_port=22,
            addresses=("10.42.0.21",),
        )

    clock.now = 2000.0
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
        ttl=30,
    )
    raw_session, _ = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )
    clock.now += JOIN_SESSION_TTL_SECONDS + 1
    with pytest.raises(EnrollmentError, match="invalid or expired"):
        store.authorize_session(raw_session)


def test_claim_session_outlives_an_allowed_worker_dependency_install(tmp_path):
    clock = _Clock()
    store = ClusterEnrollmentStore(tmp_path, clock=clock)
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
    )
    raw_session, session = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )

    clock.now += 90 * 60

    assert store.authorize_session(raw_session) == session


def test_completion_is_bound_to_claimed_worker_identity(tmp_path):
    store = ClusterEnrollmentStore(tmp_path)
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
    )
    raw_session, _ = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )

    with pytest.raises(EnrollmentError, match="identity changed"):
        store.complete(raw_session, _node(node_id="cuda-worker-2-machine"))

    completed = store.complete(raw_session, _node())
    assert completed.node_id == "cuda-worker-1-machine"
    assert store.list_nodes()[0].node_id == "cuda-worker-1-machine"
    with pytest.raises(EnrollmentError, match="invalid or expired"):
        store.authorize_session(raw_session)


def test_completed_nodes_persist_without_credentials(tmp_path):
    store = ClusterEnrollmentStore(tmp_path)
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
    )
    raw_session, _ = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )
    store.complete(raw_session, _node())

    restored = ClusterEnrollmentStore(tmp_path)
    serialized = store.path.read_text(encoding="utf-8")

    assert restored.list_nodes() == (_node(),)
    assert raw_key not in serialized
    assert raw_session not in serialized
    assert "join_key" not in serialized
    assert "session_token" not in serialized
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_revocation_invalidates_a_claim_session(tmp_path):
    store = ClusterEnrollmentStore(tmp_path)
    raw_key, issued = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
    )
    raw_session, _ = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )

    assert store.revoke_join_key(issued["join_id"]) is True
    with pytest.raises(EnrollmentError, match="invalid or expired"):
        store.authorize_session(raw_session)


def test_claim_session_can_still_be_revoked_after_join_key_expiry(tmp_path):
    clock = _Clock()
    store = ClusterEnrollmentStore(tmp_path, clock=clock)
    raw_key, issued = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="a" * 64,
        ttl=30,
    )
    raw_session, _ = store.claim(
        raw_key,
        node_id="cuda-worker-1-machine",
        hostname="cuda-worker-1",
        ssh_user="omlxworker",
        ssh_port=22,
        addresses=("10.42.0.21",),
    )
    clock.now += 31

    assert store.to_dict()["join_keys"][0]["status"] == "used"
    assert store.revoke_join_key(issued["join_id"]) is True
    with pytest.raises(EnrollmentError, match="invalid or expired"):
        store.authorize_session(raw_session)
