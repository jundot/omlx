# SPDX-License-Identifier: Apache-2.0
"""The plan the user approves must be the plan that launches.

Every defect these tests cover survived a green suite, because the suite called
the planner directly with a ``role=`` and a ``max_weight_bytes=`` the launch
path never supplied. So these go the other way round: they start from the
payload the dashboard's own JavaScript builds, post it to the real router, and
finish at the ``--plan`` argument ``mlx.launch`` would hand each rank — the one
channel a per-node setting can actually travel on.

Measured before the fix, on the audit's scenario (107.5 GiB MacBook marked
Workstation with a 40 GiB split cap, 60-layer 241 GiB model)::

    approved   mbp layers 51-60   37.0 GiB planned   53.75 GiB held back
    launched   mbp layers 36-60   97.0 GiB planned    8.00 GiB held back

The 97 GiB stage is what the auto-tune re-plan built, on a Mac the user had
capped at 40 and marked as one they were working on.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes

GiB = 1024**3
_REPO = Path(__file__).resolve().parents[1]
_DASHBOARD_JS = _REPO / "omlx" / "admin" / "static" / "js" / "dashboard.js"
_CLUSTER_HTML = _REPO / "omlx" / "admin" / "templates" / "dashboard" / "_cluster.html"

# The two Macs from the incident. Rank 0 is the local coordinator, which on the
# dashboard is always the Mac the browser is on — the laptop.
_MBP_CAPACITY = int(107.5 * GiB)
_STUDIO_CAPACITY = 512 * GiB
_SPLIT_CAP_GIB = 40
# What the dashboard actually offers: a reserve slider that defaults low, and a
# Workstation button beside it. The reserve is the value the role used to be
# silenced by.
_TYPED_RESERVE_GIB = 8
_WORKSTATION_RESERVE = int(53.75 * GiB)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# The dashboard's own payload builder, executed rather than eyeballed.
# ---------------------------------------------------------------------------


def _method_source(name: str) -> str:
    """Lift one method out of the Alpine component so node can run it."""

    source = _DASHBOARD_JS.read_text()
    # The definition, not a call site: a method is declared at the start of a
    # line, while every caller reaches it through `this.`.
    match = re.search(rf"^[ \t]*(?:async\s+)?{re.escape(name)}\(", source, re.M)
    assert match is not None, f"dashboard.js has no {name}() method"
    start = match.start()
    cursor = source.index("(", start)
    depth = 0
    for index in range(cursor, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                cursor = index
                break
    body_start = source.index("{", cursor)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"{name} has unbalanced braces in dashboard.js")


def _run_dashboard_node_payloads(state: dict) -> list[dict]:
    """Call the shipped ``clusterNodePayloads()`` with a fake component state."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the dashboard's payload builder")
    script = (
        "const state = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        "const component = { " + _method_source("clusterNodePayloads") + " };\n"
        "console.log(JSON.stringify(component.clusterNodePayloads.call(state)));\n"
    )
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps(state),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _dashboard_state() -> dict:
    """The component state after the user sets a role, a reserve and a split."""

    return {
        "clusterPlanNodes": [
            {
                "key": 1,
                "node_id": "mbp",
                "capacity_gib": _MBP_CAPACITY / GiB,
                "reserve_gib": _TYPED_RESERVE_GIB,
                "role": "workstation",
            },
            {
                "key": 2,
                "node_id": "studio",
                "capacity_gib": _STUDIO_CAPACITY / GiB,
                "reserve_gib": _TYPED_RESERVE_GIB,
                "role": "headless",
            },
        ],
        "clusterSplitGiB": _SPLIT_CAP_GIB,
    }


def test_the_dashboards_payload_builder_carries_the_role_and_the_split_cap():
    """Run the shipped JavaScript, not a Python restatement of it.

    ``activateClusterDeployment`` used to build ``{node_id, capacity_bytes,
    reserve_bytes}`` and nothing else, so ``role`` defaulted to headless and
    ``max_weight_bytes`` to 0 — "planner, balance freely" — on the only path a
    user can reach.
    """

    payloads = _run_dashboard_node_payloads(_dashboard_state())

    assert [item["node_id"] for item in payloads] == ["mbp", "studio"]
    assert payloads[0]["role"] == "workstation"
    assert payloads[0]["max_weight_bytes"] == _SPLIT_CAP_GIB * GiB
    assert payloads[0]["reserve_bytes"] == _TYPED_RESERVE_GIB * GiB
    assert payloads[1]["role"] == "headless"


def test_preview_and_activation_build_their_node_payload_in_one_place():
    """Two builders is the defect; one builder is the fix.

    Static, deliberately: the failure mode is a *second* construction site
    appearing, which no amount of exercising the first one would catch.
    """

    source = _DASHBOARD_JS.read_text()
    for name in ("runClusterPlan", "activateClusterDeployment"):
        body = _method_source(name)
        assert "this.clusterNodePayloads(" in body, (
            f"{name} must build its node payload with clusterNodePayloads()"
        )
        assert "capacity_bytes:" not in body, (
            f"{name} builds a node payload inline again — that is the drift "
            f"that dropped role and max_weight_bytes at activation"
        )
    # One definition, several callers.
    assert len(re.findall(r"clusterNodePayloads\(\{?", source)) >= 4
    assert source.count("clusterNodePayloads({ validate = false } = {})") == 1


def test_the_plan_request_names_the_parallelism_it_is_planning():
    """/plan defaulted ``tensor_parallel_size`` to 1 while activation sent 2.

    Different planner branch, different layer ranges, different plan hash — and
    the staleness guard was satisfied because it already included the value the
    request omitted.
    """

    body = _method_source("runClusterPlan")
    assert "tensor_parallel_size: Number(this.clusterPlanTensorParallelSize)" in body


def test_soft_weight_target_is_clamped_to_current_safe_budget():
    """A role change after dragging a slider must replan, not reject."""

    from omlx.cluster import routes

    gib = 1024**3
    request = routes.ClusterPlanNodeRequest(
        node_id="MacBook Pro",
        capacity_bytes=64 * gib,
        reserve_bytes=8 * gib,
        role="workstation",
        target_weight_bytes=63 * gib,
    )

    budget = routes._node_budgets([request])[0]

    assert budget.target_weight_bytes == budget.usable_bytes


# ---------------------------------------------------------------------------
# The server: same budgets in the preview, the deployment and the re-plan.
# ---------------------------------------------------------------------------


def _layout(path: str):
    from omlx.cluster.planner import ModelLayout

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GiB,
        layer_weight_bytes=(4 * GiB,) * 60,
        supports_pipeline=True,
    )


def _profile(node_id: str, rank: int, decode: float) -> dict:
    return {
        "node_id": node_id,
        "rank": rank,
        "decode_weight_bytes_per_second": decode,
        "prefill_weight_bytes_per_second": decode,
        "collective_latency_seconds": 0.001,
        "collective_bandwidth_bytes_per_second": 40e9,
        "backend": "ring",
        "measured_at": "2026-07-28T00:00:00+00:00",
        "samples": 5,
        "source": "synthetic_mlx_probe",
    }


def _nodes() -> list[dict]:
    """The payload the dashboard posts, in both places, after the fix."""

    return [
        {
            "node_id": "mbp",
            "capacity_bytes": _MBP_CAPACITY,
            "reserve_bytes": _TYPED_RESERVE_GIB * GiB,
            "role": "workstation",
            "max_weight_bytes": _SPLIT_CAP_GIB * GiB,
        },
        {
            "node_id": "studio",
            "capacity_bytes": _STUDIO_CAPACITY,
            "reserve_bytes": _TYPED_RESERVE_GIB * GiB,
            "role": "headless",
        },
    ]


def _hosts() -> list[dict]:
    return [
        {"node_id": "mbp", "ssh": "127.0.0.1", "ips": ["10.0.0.1"]},
        {"node_id": "studio", "ssh": "studio.local", "ips": ["10.0.0.2"]},
    ]


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """A router whose only fakes are the things that touch another Mac."""

    from omlx.cluster.registry import configure_cluster_registry

    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "big"
    model_path.mkdir(parents=True)
    monkeypatch.setattr(routes, "inspect_safetensors_layout", _layout)
    monkeypatch.setattr(routes, "check_peers", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        routes,
        "preflight_remote_hosts",
        lambda deployment: [{"rank": rank} for rank in range(deployment.world_size)],
    )

    class ReadyEngine:
        def __init__(self, deployment):
            self.deployment = deployment

        async def generate(self, *_args, **_kwargs):
            return SimpleNamespace(completion_tokens=1)

        def cluster_status(self):
            return {"phase": "ready", "ranks": []}

    class ReadyPool:
        def __init__(self):
            self.entry = SimpleNamespace(engine=None)

        def resolve_cluster_model_id(self, path):
            assert path == str(model_path)
            return "big"

        def get_entry(self, model_id):
            assert model_id == "big"
            return self.entry

        async def prepare_cluster_reload(self, model_id):
            assert model_id == "big"
            self.entry.engine = None

        async def get_engine(self, model_id):
            assert model_id == "big"
            deployment = routes.get_cluster_registry().get_for_model(str(model_path))
            self.entry.engine = ReadyEngine(deployment)
            return self.entry.engine

    pool = ReadyPool()
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    return model_path


def _activate(model_path, *, auto_tune: bool, approved_placement: str = "", **extra):
    if not approved_placement:
        approved_placement = _preview(model_path)["placement_signature"]
    body = {
        "deployment_id": "approval-test",
        "model_path": str(model_path),
        "backend": "ring",
        "nodes": _nodes(),
        "hosts": _hosts(),
        "auto_tune": auto_tune,
        "approved_placement": approved_placement,
    }
    body.update(extra)
    return _client().post("/admin/api/cluster/deployments", json=body)


def _preview(model_path) -> dict:
    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": str(model_path),
            "nodes": _nodes(),
            "tensor_parallel_size": 1,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _by_node(plan: dict) -> dict[str, dict]:
    return {item["node_id"]: item for item in plan["assignments"]}


def test_a_role_raises_a_reserve_and_can_never_be_silenced_by_one(cluster):
    """The one case the UI produces: an explicit reserve *and* a role.

    ``if not reserve_bytes and node.role`` meant the Workstation button did
    nothing whenever the reserve field held a number, and the dashboard always
    sends one. Measured on the real function before the fix: reserve 400 MB +
    role workstation resolved to 400 MB.
    """

    plan = _preview(cluster)
    mbp = _by_node(plan)["mbp"]

    assert mbp["role"] == "workstation"
    assert mbp["reserve_bytes"] == _WORKSTATION_RESERVE
    assert mbp["reserve_bytes"] > _TYPED_RESERVE_GIB * GiB
    # ...and a role never lowers a reserve the caller deliberately raised.
    generous = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": str(cluster),
            "nodes": [
                dict(_nodes()[0], reserve_bytes=80 * GiB, max_weight_bytes=0),
                _nodes()[1],
            ],
        },
    )
    assert generous.status_code == 200, generous.text
    assert _by_node(generous.json())["mbp"]["reserve_bytes"] == 80 * GiB


def test_a_manual_memory_slider_replaces_the_automatic_role_default(cluster):
    """The number shown beside the slider must be the number planning uses."""

    manual_reserve = 18 * GiB
    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": str(cluster),
            "nodes": [
                dict(
                    _nodes()[0],
                    reserve_bytes=manual_reserve,
                    manual_memory_limit=True,
                    max_weight_bytes=0,
                ),
                _nodes()[1],
            ],
        },
    )

    assert response.status_code == 200, response.text
    mbp = _by_node(response.json())["mbp"]
    assert mbp["role"] == "workstation"
    assert mbp["manual_memory_limit"] is True
    assert mbp["reserve_bytes"] == manual_reserve


def test_preview_and_activation_produce_the_same_plan(cluster):
    """The seam: /plan and /deployments must plan the same thing.

    Before the fix they did not even take the same planner branch, and the
    activation payload carried neither the role nor the cap.
    """

    preview = _preview(cluster)
    response = _activate(cluster, auto_tune=False)
    assert response.status_code == 200, response.text
    launched = response.json()["plan"]

    assert launched["plan_hash"] == preview["plan_hash"]
    assert launched["placement_signature"] == preview["placement_signature"]
    assert response.json()["plan_changes"]["changed"] is False


def test_the_role_and_the_cap_reach_the_rank_through_the_launch_argv(cluster):
    """End of the line: the argument vector ``mlx.launch`` ships to each Mac.

    ``build_mlx_launch_argv`` emits one argv every host runs identically, so a
    flag cannot say "studio=headless, macbook=workstation". The encoded plan is
    indexed by rank on arrival, so this is the value the rank sizes its own
    admission from. Nothing here starts a process.
    """

    from omlx.cluster.deployment import ClusterDeployment, decode_worker_contract
    from omlx.cluster.launch import build_mlx_launch_argv

    response = _activate(cluster, auto_tune=False)
    assert response.status_code == 200, response.text

    deployment = ClusterDeployment.from_dict(response.json()["deployment"])
    argv = build_mlx_launch_argv(
        deployment,
        hostfile=Path("/tmp/omlx-approval-test-hostfile.json"),
        api_port=8080,
        collective_port=9090,
    )
    encoded = argv[argv.index("--plan") + 1]
    _hash, assignments, _profiles, _tp = decode_worker_contract(encoded)
    by_rank = {item.rank: item for item in assignments}

    assert by_rank[0].node_id == "mbp"
    assert by_rank[0].role == "workstation"
    assert by_rank[0].reserve_bytes == _WORKSTATION_RESERVE
    assert by_rank[0].planned_weight_bytes <= _SPLIT_CAP_GIB * GiB
    assert by_rank[1].role == "headless"


def test_auto_tuning_replans_from_the_budgets_the_user_approved(cluster, monkeypatch):
    """The re-plan that discarded the reserve and the cap.

    ``_performance_optimized_deployment`` was the module's third ``NodeBudget``
    construction site and the only one with neither. With a probe that reports
    the laptop as the faster Mac, the unconstrained planner puts 97.0 GiB on a
    107.5 GiB machine; the constrained one may not exceed the 40 GiB cap.
    """

    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda deployment: {
            "ok": True,
            "profiles": [_profile("mbp", 0, 60e9), _profile("studio", 1, 20e9)],
        },
    )

    response = _activate(cluster, auto_tune=True)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["performance_probe"]["status"] == "placement_locked"
    assert payload["plan"]["placement_signature"] == _preview(cluster)[
        "placement_signature"
    ]

    mbp = _by_node(payload["plan"])["mbp"]
    assert mbp["role"] == "workstation"
    assert mbp["reserve_bytes"] == _WORKSTATION_RESERVE
    assert mbp["planned_weight_bytes"] <= _SPLIT_CAP_GIB * GiB

    # The same probe, planned without the cap and the role, is what used to be
    # persisted. Asserting it here keeps the test pinned to the real regression
    # rather than to an arithmetic identity.
    from omlx.cluster.performance import NodePerformanceProfile
    from omlx.cluster.planner import NodeBudget, plan_unequal_pipeline

    unconstrained = plan_unequal_pipeline(
        _layout(str(cluster)),
        [
            NodeBudget(
                node_id="mbp",
                capacity_bytes=_MBP_CAPACITY,
                reserve_bytes=_TYPED_RESERVE_GIB * GiB,
                rank=0,
                performance=NodePerformanceProfile.from_dict(_profile("mbp", 0, 60e9)),
            ),
            NodeBudget(
                node_id="studio",
                capacity_bytes=_STUDIO_CAPACITY,
                reserve_bytes=_TYPED_RESERVE_GIB * GiB,
                rank=1,
                performance=NodePerformanceProfile.from_dict(
                    _profile("studio", 1, 20e9)
                ),
            ),
        ],
        microbatch_size=4,
    )
    dropped = {item.node_id: item for item in unconstrained.assignments}["mbp"]
    assert dropped.planned_weight_bytes > 90 * GiB
    assert mbp["planned_weight_bytes"] < dropped.planned_weight_bytes


def test_a_replan_that_moves_layers_is_reported_and_not_applied(
    cluster, monkeypatch
):
    """Tuning may re-plan. It may not do it behind the approval."""

    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda deployment: {
            "ok": True,
            "profiles": [_profile("mbp", 0, 60e9), _profile("studio", 1, 20e9)],
        },
    )

    preview = _preview(cluster)
    response = _activate(cluster, auto_tune=True)
    assert response.status_code == 200, response.text
    changes = response.json()["plan_changes"]

    assert changes["changed"] is True
    assert changes["approved_signature"] == preview["placement_signature"]
    assert changes["launched_signature"] == preview["placement_signature"]
    assert response.json()["plan"]["placement_signature"] == preview[
        "placement_signature"
    ]
    assert response.json()["performance_probe"]["status"] == "placement_locked"
    moved = {item["node_id"] for item in changes["ranks"]}
    assert "mbp" in moved
    summary = next(item for item in changes["ranks"] if item["node_id"] == "mbp")
    assert "would hold layers" in summary["summary"]
    assert summary["layer_delta"] != 0


def test_precomputed_profiles_skip_the_post_staging_probe(cluster, monkeypatch):
    """One-click calibration is signed into the plan and never run twice."""

    profiled_nodes = [
        dict(_nodes()[0], performance=_profile("mbp", 0, 20e9)),
        dict(_nodes()[1], performance=_profile("studio", 1, 60e9)),
    ]
    preview_response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": str(cluster),
            "nodes": profiled_nodes,
            "tensor_parallel_size": 1,
        },
    )
    assert preview_response.status_code == 200, preview_response.text
    preview = preview_response.json()
    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda _deployment: (_ for _ in ()).throw(
            AssertionError("the signed pre-staging measurement must be reused")
        ),
    )

    response = _activate(
        cluster,
        auto_tune=True,
        nodes=profiled_nodes,
        approved_placement=preview["placement_signature"],
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["performance_probe"]["status"] == "precomputed_before_staging"
    assert len(payload["deployment"]["performance_profiles"]) == 2
    assert payload["plan"]["placement_signature"] == preview["placement_signature"]


def test_activation_refuses_a_plan_that_is_not_the_one_that_was_approved(cluster):
    """The guard that makes "approved" a fact rather than a hope.

    Posting the preview's signature alongside a payload whose role has been
    dropped is exactly what the dashboard used to do by accident.
    """

    preview = _preview(cluster)
    drifted = [dict(_nodes()[0], role="headless", max_weight_bytes=0), _nodes()[1]]

    response = _client().post(
        "/admin/api/cluster/deployments",
        json={
            "deployment_id": "approval-test",
            "model_path": str(cluster),
            "backend": "ring",
            "nodes": drifted,
            "hosts": _hosts(),
            "auto_tune": False,
            "approved_placement": preview["placement_signature"],
        },
    )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "not the plan you approved" in detail
    # The refusal names what it would have launched instead.
    assert "mbp layers" in detail

    # And nothing was registered.
    listed = _client().get("/admin/api/cluster/deployments")
    assert listed.json()["deployments"] == []


def test_the_approved_plan_activates_when_it_still_matches(cluster):
    response = _activate(
        cluster,
        auto_tune=False,
        approved_placement=_preview(cluster)["placement_signature"],
    )
    assert response.status_code == 200, response.text


def test_the_approval_signature_ignores_tuning_that_moves_no_layer(cluster):
    """Why the guard is built on the placement and not on ``plan_hash``.

    ``tune_execution_settings`` lowers the pipeline microbatch on a tight plan
    and ``_create_deployment`` then re-plans, which changes the hash without
    moving a single layer. A guard keyed on the hash would refuse those
    activations with a reason nothing on the page could explain.
    """

    from omlx.cluster.planner import plan_unequal_pipeline

    budgets = routes._node_budgets(
        [routes.ClusterPlanNodeRequest(**node) for node in _nodes()]
    )
    model = _layout(str(cluster))
    coarse = plan_unequal_pipeline(model, budgets, microbatch_size=4).to_dict()
    fine = plan_unequal_pipeline(model, budgets, microbatch_size=1).to_dict()

    assert coarse["plan_hash"] != fine["plan_hash"]
    assert routes._placement_signature(coarse) == routes._placement_signature(fine)


def test_an_approved_plan_survives_auto_tune_when_the_probe_cannot_run(
    cluster, monkeypatch
):
    """The memory fallback still has to be the plan that was approved."""

    from omlx.cluster.launch import DistributedLaunchError

    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda deployment: (_ for _ in ()).throw(
            DistributedLaunchError("benchmark link unavailable")
        ),
    )

    preview = _preview(cluster)
    response = _activate(
        cluster, auto_tune=True, approved_placement=preview["placement_signature"]
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["performance_probe"]["status"] == "memory_fallback"
    assert payload["plan"]["placement_signature"] == preview["placement_signature"]
    assert payload["plan_changes"]["changed"] is False


def test_the_catalogue_answers_with_the_same_budgets_as_the_planner(cluster):
    """ "Will this model run?" must be asked of the plan that would run.

    The catalogue built its own ``NodeBudget`` too, without the role, so a
    model it called runnable could be one the approved plan refuses.
    """

    response = _client().post(
        "/admin/api/cluster/catalogue",
        json={"nodes": _nodes(), "model_paths": [str(cluster)]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    # capacity is untouched by a reserve; the reserve shows up as the model
    # verdict, which must agree with the plan the same nodes produce.
    assert payload["cluster_capacity_bytes"] == _MBP_CAPACITY + _STUDIO_CAPACITY


# ---------------------------------------------------------------------------
# The template surfaces that make the difference visible.
# ---------------------------------------------------------------------------


def test_the_activation_panel_shows_what_a_replan_moved():
    html = _CLUSTER_HTML.read_text()

    assert "data-cluster-plan-changes" in html
    assert "clusterPlanChanges.ranks" in html
    assert "clusterPlanIsStale()" in html
    # The reserve a role sets is read back off the plan, not off local state.
    assert "clusterGiB(item.reserve_bytes)" in html

    javascript = _DASHBOARD_JS.read_text()
    assert "clusterPlanChanges: null," in javascript
    assert (
        "this.clusterPlanChanges = this.clusterActivationResult.plan_changes"
        in javascript
    )
    assert "approved_placement: this.clusterPlan?.placement_signature" in javascript
    assert "clusterPlanIsStale()" in javascript


def test_the_cluster_template_tags_balance():
    """An unclosed div swallows every panel below it and renders green."""

    from html.parser import HTMLParser

    void = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    class Balance(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.stack: list[tuple[str, tuple[int, int]]] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append((tag, self.getpos()))

        def handle_startendtag(self, tag, attrs):
            return

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack:
                self.errors.append(f"</{tag}> at {self.getpos()} closes nothing")
                return
            opened, position = self.stack.pop()
            if opened != tag:
                self.errors.append(
                    f"</{tag}> at {self.getpos()} closes <{opened}> opened at {position}"
                )

    parser = Balance()
    parser.feed(_CLUSTER_HTML.read_text())
    parser.close()
    errors = parser.errors + [
        f"<{tag}> opened at {position} is never closed"
        for tag, position in parser.stack
    ]
    assert not errors, errors


def test_every_alpine_expression_in_the_cluster_template_parses_as_javascript():
    """A syntax error in an Alpine attribute is silent until someone opens the tab."""

    import html as html_module

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to parse Alpine expressions")

    source = _CLUSTER_HTML.read_text()
    attribute = re.compile(
        r'(?P<name>(?:x-[a-z:.\-]+|@[A-Za-z0-9:.\-]+|:[A-Za-z0-9:.\-]+))\s*=\s*"(?P<value>[^"]*)"',
        re.S,
    )
    statement_attributes = {"x-init", "x-data", "x-effect"}
    checks = []
    for match in attribute.finditer(source):
        name = match.group("name")
        if name == "x-cloak" or name.startswith("x-transition"):
            continue
        value = html_module.unescape(match.group("value")).strip()
        if not value:
            continue
        base = name.split(".")[0].split(":")[0]
        checks.append(
            {
                "name": name,
                "line": source[: match.start()].count("\n") + 1,
                "value": value,
                "statement": base in statement_attributes or base.startswith("@"),
            }
        )
    assert len(checks) > 100, "expected the cluster template to be full of Alpine"

    script = """
const checks = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const failures = [];
for (const check of checks) {
    const body = check.statement ? check.value : `(${check.value})`;
    try {
        new Function('$event', '$el', '$refs', '$store', '$dispatch', '$nextTick', body);
    } catch (error) {
        failures.push(`${check.name} line ${check.line}: ${error.message} :: ${check.value}`);
    }
}
console.log(JSON.stringify(failures));
"""
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps(checks),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def _template_expressions(*needles: str) -> list[str]:
    """Alpine attribute values from the real template, by what they reference."""

    import html as html_module

    source = _CLUSTER_HTML.read_text()
    attribute = re.compile(
        r'(?:x-[a-z:.\-]+|@[A-Za-z0-9:.\-]+|:[A-Za-z0-9:.\-]+)\s*=\s*"(?P<value>[^"]*)"',
        re.S,
    )
    found = [
        html_module.unescape(match.group("value")).strip()
        for match in attribute.finditer(source)
        if any(needle in match.group("value") for needle in needles)
    ]
    assert found, f"no cluster template expression references {needles}"
    return found


def test_the_new_plan_expressions_survive_a_page_with_no_plan_yet():
    """`x-show` only hides — `x-text` on the same element still evaluates.

    So an expression guarded by `x-show` must be total, not merely unreachable
    when the guard is false. `clusterPlannedReserveGiB(...).toFixed(1)` threw a
    TypeError on every render before a plan existed, which takes the whole
    Alpine component down rather than blanking one line.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to evaluate Alpine expressions")

    methods = ", ".join(
        _method_source(name)
        for name in (
            "clusterNodePayloads",
            "clusterCurrentPlanSignature",
            "clusterPlanIsStale",
            "clusterPlannedReserveGiB",
            "clusterGiB",
        )
    )
    payload = {
        # The component exactly as the page starts: nothing planned, nothing
        # activated, the default two nodes present.
        "state": {
            "clusterPlan": None,
            "clusterPlanChanges": None,
            "_clusterPlanSignature": "",
            "clusterSplitGiB": None,
            "clusterPlanMode": "model",
            "clusterPlanModelPath": "",
            "clusterPlanModelSizeGiB": 0,
            "clusterPlanLayerCount": 0,
            "clusterExecutionProfile": "balanced",
            "clusterPlanTensorParallelSize": 1,
            "clusterPlanNodes": [
                {
                    "key": 1,
                    "node_id": "studio",
                    "capacity_gib": 256,
                    "reserve_gib": 8,
                    "role": "headless",
                },
                {
                    "key": 2,
                    "node_id": "mobile",
                    "capacity_gib": 128,
                    "reserve_gib": 8,
                    "role": "workstation",
                },
            ],
        },
        "expressions": _template_expressions(
            "clusterPlannedReserveGiB", "clusterPlanIsStale"
        ),
    }

    script = (
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        "const component = Object.assign({ " + methods + " }, input.state);\n"
        "const scope = new Proxy(component, {\n"
        "  has: () => true,\n"
        "  get: (target, key) => {\n"
        "    if (key === Symbol.unscopables) return undefined;\n"
        "    if (key === 'node') return target.clusterPlanNodes[0];\n"
        "    const value = target[key];\n"
        "    return typeof value === 'function' ? value.bind(target) : value;\n"
        "  },\n"
        "});\n"
        "const failures = [];\n"
        "for (const expression of input.expressions) {\n"
        "  try {\n"
        "    new Function('scope', `with (scope) { return (${expression}); }`)(scope);\n"
        "  } catch (error) {\n"
        "    failures.push(`${error.message} :: ${expression}`);\n"
        "  }\n"
        "}\n"
        "console.log(JSON.stringify(failures));\n"
    )
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_the_dashboard_javascript_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to parse dashboard.js")
    result = subprocess.run(
        [node, "--check", str(_DASHBOARD_JS)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
