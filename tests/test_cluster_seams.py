# SPDX-License-Identifier: Apache-2.0
"""Tests for the seams *between* units, where every real bug has lived so far.

Each unit in ``omlx/cluster`` has had good unit coverage throughout, and the
suite has been green through a dashboard that called endpoints with a `params`
option `fetch` ignores, a pairing token whose verifier could never succeed, a
planner nothing called, and a `shard_linear` import that resolved to nothing.
All four are invisible to a test that exercises one unit at a time.

These tests cross the boundaries instead:
  * every URL the dashboard calls exists as a route, and vice versa
  * every generate/verify and encode/decode pair round-trips
  * no public function in the package is unreachable
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CLUSTER = _REPO / "omlx" / "cluster"
_DASHBOARD_JS = _REPO / "omlx" / "admin" / "static" / "js" / "dashboard.js"

_PREFIX = "/admin/api/cluster"
# Template literals interpolate with ${...}, which may contain calls and nested
# parens: /deployments/${encodeURIComponent(id)}
_CLUSTER_URL = re.compile(
    re.escape(_PREFIX) + r"(?P<path>(?:\$\{[^{}]*(?:\([^)]*\))?[^{}]*\}|[A-Za-z0-9/_\-.])*)"
)


def _registered_routes() -> set[str]:
    from omlx.cluster import routes

    return {
        re.sub(r"\{[^{}]+\}", "{parameter}", route.path)
        for route in routes.router.routes
        if getattr(route, "path", None)
    }


def _js_called_paths() -> set[str]:
    """Cluster URLs the dashboard builds, normalised to their route shape."""

    called = set()
    for match in _CLUSTER_URL.finditer(_DASHBOARD_JS.read_text()):
        path = match.group("path").split("?")[0]
        # Any interpolated segment stands for a path parameter.
        path = re.sub(r"\$\{[^{}]*(?:\([^)]*\))?[^{}]*\}", "{parameter}", path)
        path = path.rstrip("/") if path not in ("", "/") else path
        called.add(_PREFIX + path)
    return called


def test_every_cluster_url_the_dashboard_calls_is_a_real_route():
    """A typo or a renamed endpoint here is a 404 no unit test would notice."""

    missing = _js_called_paths() - _registered_routes()
    assert not missing, (
        f"dashboard.js calls cluster endpoints that are not registered: "
        f"{sorted(missing)}"
    )


def test_no_cluster_route_is_unreachable_from_the_dashboard():
    """Every route should have a caller, or be deliberately listed here.

    A route with no caller is either dead or a feature that was never wired up —
    both worth knowing about.
    """

    allowed_without_caller: set[str] = set()
    unreachable = _registered_routes() - _js_called_paths() - allowed_without_caller
    assert not unreachable, (
        f"cluster routes nothing calls: {sorted(unreachable)} — wire them up or "
        f"add them to allowed_without_caller with a reason"
    )


def test_fetch_calls_never_use_a_params_option():
    """`fetch(url, {params})` is silently ignored; query strings must be built.

    This exact mistake made every pairing and key-exchange call return 422 while
    the suite stayed green.
    """

    source = _DASHBOARD_JS.read_text()
    offenders = []
    for index, line in enumerate(source.splitlines(), start=1):
        if re.search(r"^\s*params:\s*\{", line):
            window = "\n".join(source.splitlines()[max(0, index - 6) : index])
            if "fetch(" in window:
                offenders.append(index)
    assert not offenders, (
        f"dashboard.js:{offenders} pass `params` to fetch(); fetch ignores it — "
        f"use URLSearchParams and put it in the URL"
    )


def test_pairing_token_round_trips():
    from omlx.cluster.discovery import generate_pairing_token, verify_pairing_token

    secret = "correct-horse-battery-staple"
    assert verify_pairing_token(
        generate_pairing_token(shared_secret=secret),
        shared_secret=secret,
    ) is True


def test_pairing_token_rejects_a_tampered_payload():
    import base64
    import json

    from omlx.cluster.discovery import generate_pairing_token, verify_pairing_token

    secret = "correct-horse-battery-staple"
    payload = json.loads(
        base64.urlsafe_b64decode(generate_pairing_token(shared_secret=secret))
    )
    payload["token"] = "substituted"
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    assert verify_pairing_token(forged, shared_secret=secret) is False


def test_worker_contract_round_trips_with_tensor_parallelism():
    """Encode/decode must preserve TP, and the hash must notice a change."""

    from omlx.cluster.deployment import decode_worker_contract

    planner = pytest.importorskip("omlx.cluster.planner")
    model = planner.ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 8,
        tensor_parallel_heads=16,
    )
    nodes = [
        planner.NodeBudget(
            node_id=f"node-{index}",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=index,
        )
        for index in range(4)
    ]
    plan = planner.plan_hybrid(model, nodes, tensor_parallel_size=2)

    from omlx.cluster.deployment import ClusterDeployment, ClusterHost

    deployment = ClusterDeployment(
        deployment_id="seam-test",
        model="/models/test",
        backend="ring",
        hosts=(
            # Rank 0 is the local coordinator by contract.
            ClusterHost(node_id="node-0", ssh="127.0.0.1", ips=("10.0.0.1",)),
            *(
                ClusterHost(
                    node_id=f"node-{index}",
                    ssh=f"node{index}.local",
                    ips=(f"10.0.0.{index + 1}",),
                )
                for index in range(1, 4)
            ),
        ),
        assignments=plan.assignments,
        plan_hash=plan.plan_hash,
        tensor_parallel_size=plan.tensor_parallel_size,
    )

    plan_hash, assignments, _profiles, tp_size = decode_worker_contract(
        deployment.encode_worker_plan()
    )

    assert plan_hash == plan.plan_hash
    assert tp_size == plan.tensor_parallel_size == 2
    assert [item.rank for item in assignments] == [0, 1, 2, 3]
    assert [item.tensor_parallel_rank for item in assignments] == [0, 1, 0, 1]

    # Same inputs, different TP degree -> different hash, so a stale plan cannot
    # silently launch against a different topology.
    other = planner.plan_hybrid(model, nodes, tensor_parallel_size=1)
    assert other.plan_hash != plan.plan_hash


def _public_functions(path: Path) -> list[str]:
    """Module-level functions, excluding ones a framework calls by decorator.

    FastAPI route handlers are referenced only by ``@router.get(...)``, so a
    plain name search would always call them dead.
    """

    tree = ast.parse(path.read_text())
    names = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.decorator_list:
            continue
        names.append(node.name)
    return names


def test_no_unreachable_functions_in_the_cluster_package():
    """A function with no non-test caller is not a finished feature.

    `plan_tensor_parallel` was written, tested and ticked off while nothing in
    the product called it.
    """

    allowed_uncalled = {
        ("deployment.py", "decode_worker_plan"),
        # Maintainer-only real-collective regression gate. It deliberately is
        # not exposed in the GUI or production route surface.
        ("collective.py", "_run_local_minimax_decode_smoke"),
        # known_hosts helpers, not yet wired into the pairing flow.
        ("ssh_keys.py", "add_host_key"),
        ("ssh_keys.py", "_ssh_executable"),
        # Cache maintenance, used by tests and available to callers that know
        # the topology changed.
        ("discovery.py", "clear_peer_transport_cache"),
        # Link discovery, exposed ahead of the hostfile builder that will
        # replace its hand-typed addresses and its empty ClusterHost.rdma.
        ("transport.py", "resolve_link_addresses"),
        ("autoconfigure.py", "build_rdma_matrix"),
        # Peer import preflight, exposed ahead of the /autoconfigure handler
        # that will call it alongside preflight_issues.
        ("autoconfigure.py", "peer_import_issues"),
    }

    sources = {
        path: path.read_text() for path in (_REPO / "omlx").rglob("*.py")
    }

    uncalled = []
    for path in sorted(_CLUSTER.glob("*.py")):
        for name in _public_functions(path):
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            hits = sum(len(pattern.findall(text)) for text in sources.values())
            if hits <= 1 and (path.name, name) not in allowed_uncalled:
                uncalled.append(f"{path.name}:{name}")

    assert not uncalled, (
        f"unreachable functions: {uncalled} — give them a caller, delete them, "
        f"or add them to allowed_uncalled with a reason"
    )


def test_every_literal_ssh_and_scp_command_uses_the_shared_policy():
    """One raw subprocess is enough to bring an interactive prompt back."""

    offenders = []
    for path in sorted(_CLUSTER.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.List) or not node.elts:
                continue
            first = node.elts[0]
            if not isinstance(first, ast.Constant) or first.value not in {"ssh", "scp"}:
                continue
            protected = any(
                isinstance(item, ast.Starred)
                and isinstance(item.value, ast.Call)
                and isinstance(item.value.func, ast.Name)
                and item.value.func.id == "cluster_ssh_options"
                for item in node.elts
            )
            if not protected:
                offenders.append((path.name, node.lineno, first.value))

    assert not offenders, f"SSH/SCP commands bypass shared policy: {offenders}"


def test_discovery_does_not_import_the_transport_prober():
    """Peer listing must not sit behind an SSH round trip.

    Inlining `detect_transports` here hung the suite; a daemon thread then kept
    mutating the peer list after the call returned.
    """

    source = (_CLUSTER / "discovery.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "transport":
            pytest.fail(
                "discovery.py imports .transport; transport probing must stay "
                "off the discovery request path"
            )


def test_every_get_route_answers_without_a_server_error():
    """Smoke every read-only route through the real app.

    Not about the payloads — about the wiring. A route that raises on import,
    a missing dependency, or a handler signature FastAPI cannot satisfy shows up
    here and nowhere in a unit test.
    """

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    app = FastAPI()
    app.include_router(routes.router)

    # Query args each GET needs; anything else is called bare.
    query = {
        "/admin/api/cluster/transports": {"hosts": "127.0.0.1"},
    }

    checked = 0
    with TestClient(app) as client:
        for route in routes.router.routes:
            if "GET" not in getattr(route, "methods", set()):
                continue
            if "{" in route.path:  # needs a real deployment id
                continue
            response = client.get(route.path, params=query.get(route.path, {}))
            # 503 is a legitimate "not configured on this host" answer; only a
            # 500 means the handler itself is mis-wired.
            assert response.status_code != 500, (
                f"GET {route.path} returned {response.status_code}: {response.text[:200]}"
            )
            checked += 1
    assert checked >= 5, "expected to smoke several GET routes"


def test_post_routes_reject_a_bad_body_rather_than_crashing():
    """A 422 means the contract is wired; a 500 means the handler is broken."""

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    app = FastAPI()
    app.include_router(routes.router)

    # Endpoints that act on the machine rather than validate a body. Poking
    # these blindly is not a smoke test — /ssh-key/generate overwrites the
    # user's cluster SSH key, and the smoke runners spawn real processes.
    side_effecting = {
        "/admin/api/cluster/ssh-key/generate",
        "/admin/api/cluster/ssh-key/store-keychain",
        "/admin/api/cluster/ssh-key/exchange",
        "/admin/api/cluster/ssh-key/exchange-token",
        "/admin/api/cluster/peer-probe",
        "/admin/api/cluster/worker-smoke",
        "/admin/api/cluster/collective-smoke",
        "/admin/api/cluster/pipeline-smoke",
        "/admin/api/cluster/deployments",
        "/admin/api/cluster/pairing-token",
        "/admin/api/cluster/verify-pairing-token",
    }

    permissive = {"/admin/api/cluster/guidance"}

    checked = 0
    with TestClient(app) as client:
        for route in routes.router.routes:
            if "POST" not in getattr(route, "methods", set()) or "{" in route.path:
                continue
            if route.path in side_effecting:
                continue
            if route.path in permissive:
                # Explaining a failure must never itself fail — every field is
                # optional and unknown keys are ignored by design.
                assert client.post(route.path, json={"x": 1}).status_code == 200
                checked += 1
                continue
            response = client.post(route.path, json={"deliberately": "invalid"})
            assert response.status_code == 422, (
                f"POST {route.path} should reject an invalid body with 422, "
                f"got {response.status_code}: {response.text[:200]}"
            )
            checked += 1
    assert checked >= 1


def test_key_exchange_token_round_trips():
    """create_key_exchange_token -> verify_key_exchange_token.

    The other half of the pairing flow. Its sibling (the pairing token) was
    broken in exactly this seam: a generator and a verifier that could never
    agree, each fine in isolation.
    """

    import base64

    from omlx.cluster import ssh_keys

    # The fingerprint helper base64-decodes the key blob, so it must be valid.
    blob = base64.b64encode(b"\x00" * 32).decode()
    public_key = f"ssh-ed25519 {blob} omlx"
    token = ssh_keys.create_key_exchange_token(
        public_key=public_key,
        node_id="peer-mac",
        shared_secret="correct-horse-battery-staple",
    )
    assert isinstance(token, str) and token

    decoded = ssh_keys.verify_key_exchange_token(
        token,
        shared_secret="correct-horse-battery-staple",
    )
    assert decoded is not None, "a freshly generated token must verify"
    assert decoded.node_id == "peer-mac"
    assert decoded.public_key == public_key


def test_key_exchange_rejects_a_tampered_token():
    import base64
    import json

    from omlx.cluster import ssh_keys

    blob = base64.b64encode(b"\x00" * 32).decode()
    token = ssh_keys.create_key_exchange_token(
        public_key=f"ssh-ed25519 {blob} omlx",
        node_id="peer-mac",
        shared_secret="correct-horse-battery-staple",
    )
    payload = json.loads(base64.urlsafe_b64decode(token))
    payload["node_id"] = "attacker-mac"
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

    assert ssh_keys.verify_key_exchange_token(
        forged,
        shared_secret="correct-horse-battery-staple",
    ) is None


def test_key_exchange_rejects_the_wrong_shared_secret():
    import base64

    from omlx.cluster import ssh_keys

    blob = base64.b64encode(b"\x00" * 32).decode()
    token = ssh_keys.create_key_exchange_token(
        public_key=f"ssh-ed25519 {blob} omlx",
        node_id="peer-mac",
        shared_secret="correct-horse-battery-staple",
    )

    assert ssh_keys.verify_key_exchange_token(
        token,
        shared_secret="a-different-shared-secret",
    ) is None


def test_key_exchange_rejects_an_authenticated_ssh_option_target():
    import base64
    import hashlib
    import hmac
    import json

    from omlx.cluster import ssh_keys

    secret = "correct-horse-battery-staple"
    blob = base64.b64encode(b"\x00" * 32).decode()
    token = ssh_keys.create_key_exchange_token(
        public_key=f"ssh-ed25519 {blob} omlx",
        node_id="peer-mac",
        shared_secret=secret,
    )
    payload = json.loads(base64.urlsafe_b64decode(token))
    payload["node_id"] = "-oProxyCommand"
    signed = {
        key: payload[key]
        for key in (
            "token",
            "public_key",
            "fingerprint",
            "node_id",
            "created_at",
            "expires_at",
        )
    }
    payload["signature"] = hmac.new(
        secret.encode(),
        json.dumps(signed, sort_keys=True).encode(),
        hashlib.sha256,
    ).hexdigest()
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

    assert ssh_keys.verify_key_exchange_token(
        forged,
        shared_secret=secret,
    ) is None
