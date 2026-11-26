# SPDX-License-Identifier: Apache-2.0
"""Tests for the seams *between* units, where every real bug has lived so far.

Each unit in ``omlx/cluster`` has had good unit coverage throughout, and the
suite has been green through a dashboard that called endpoints with a `params`
option `fetch` ignores, a pairing token whose verifier could never succeed, a
planner nothing called, and a `shard_linear` import that resolved to nothing.
All four are invisible to a test that exercises one unit at a time.

These tests cross the boundaries instead:
  * every generate/verify and encode/decode pair round-trips
  * no public function in the package is unreachable

The dashboard URL checks live in apps/omlx-web/tests/test_cluster_seams_ui.py.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CLUSTER = _REPO / "omlx" / "cluster"

# Template literals interpolate with ${...}, which may contain calls and nested
# parens: /deployments/${encodeURIComponent(id)}


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
        # Maintainer-only real-collective regression gate. It deliberately is
        # not exposed in the GUI or production route surface.
        ("collective.py", "_run_local_minimax_decode_smoke"),
        # Test hooks that drop process-wide cluster singletons between cases; only
        # the test suite calls them (production swaps via configure_*).
        ("identity.py", "reset_configured_identity"),
        ("registry.py", "reset_configured_device_registry"),
        ("pairing.py", "reset_pairing_manager"),
        ("pairing_routes.py", "set_pairing_manager_getter"),
    }

    name_counts = Counter(
        name
        for path in (_REPO / "omlx").rglob("*.py")
        for name in re.findall(r"\w+", path.read_text())
    )

    uncalled = []
    for path in sorted(_CLUSTER.rglob("*.py")):
        for name in _public_functions(path):
            hits = name_counts[name]
            if hits <= 1 and (path.name, name) not in allowed_uncalled:
                uncalled.append(f"{path.name}:{name}")

    assert not uncalled, (
        f"unreachable functions: {uncalled} — give them a caller, delete them, "
        f"or add them to allowed_uncalled with a reason"
    )


def test_every_literal_ssh_and_scp_command_uses_the_shared_policy():
    """One raw subprocess is enough to bring an interactive prompt back."""

    offenders = []
    for path in sorted(_CLUSTER.rglob("*.py")):
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


def test_every_get_route_answers_without_a_server_error(cluster_home):
    """Smoke every GET route through the real app.

    Not about the payloads — about the wiring. A route that raises on import,
    a missing dependency, or a handler signature FastAPI cannot satisfy shows up
    here and nowhere in a unit test.
    """

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    app = FastAPI()
    app.include_router(routes.router)

    checked = 0
    with TestClient(app) as client:
        for route in routes.router.routes:
            if "GET" not in getattr(route, "methods", set()):
                continue
            if "{" in route.path:  # needs a real deployment id
                continue
            response = client.get(route.path)
            # 503 is a legitimate "not configured on this host" answer; only a
            # 500 means the handler itself is mis-wired.
            assert (
                response.status_code != 500
            ), f"GET {route.path} returned {response.status_code}: {response.text[:200]}"
            checked += 1
    assert checked >= 3, "expected to smoke several GET routes"


def test_post_routes_reject_a_bad_body_rather_than_crashing():
    """A 422 means the contract is wired; a 500 means the handler is broken."""

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    app = FastAPI()
    app.include_router(routes.router)

    # Endpoints that act on the machine rather than validate a body. Poking
    # these blindly is not a smoke test: they probe peers over SSH or launch
    # ranks.
    side_effecting = {
        "/admin/api/cluster/peer-probe",
        "/admin/api/cluster/deployments",
    }

    checked = 0
    with TestClient(app) as client:
        for route in routes.router.routes:
            if "POST" not in getattr(route, "methods", set()) or "{" in route.path:
                continue
            if route.path in side_effecting:
                continue
            response = client.post(route.path, json={"deliberately": "invalid"})
            assert response.status_code == 422, (
                f"POST {route.path} should reject an invalid body with 422, "
                f"got {response.status_code}: {response.text[:200]}"
            )
            checked += 1
    assert checked >= 1


