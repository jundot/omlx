# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for the Cluster tab's headline status (#2703, #2704).

``runtime_compatible`` is a verdict produced by a probe that has already
finished. Folding it into the same boolean as the in-flight loading flags meant
a worker with any runtime problem sat on "Checking the connection…" forever,
while the peer card directly below it rendered the mismatch in red.

These execute the shipped ``clusterQuickStatus`` under node rather than
asserting on source text, so they fail if the behaviour regresses by any route.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_JS = ROOT / "omlx" / "admin" / "static" / "js" / "dashboard.js"

_LIFTED = ("clusterQuickStatus",)


def _method_source(name: str) -> str:
    """Lift one complete Alpine method out of the shipped dashboard source."""

    source = DASHBOARD_JS.read_text()
    match = re.search(rf"^[ \t]*(?:async\s+)?{re.escape(name)}\(", source, re.M)
    assert match is not None, f"dashboard.js has no {name}() method"
    start = match.start()
    body_start = source.index("{", source.index(")", source.index("(", start)))
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"{name} has unbalanced braces in dashboard.js")


# Every seam the status computation reads, pinned to a healthy cluster that has
# already chosen a model. Stated explicitly rather than left undefined, so a
# branch is skipped because the state says so and not because a typo made it
# falsy. clusterSelectedModel returns a model on purpose: with it null the
# healthy path stops at 'model' and the 'ready' branch is never reached.
_STUBS = """
  clusterStatus: { running: true }, clusterDeployments: [],
  clusterAutoconfigureError: '', clusterError: '', clusterDeploymentsError: '',
  clusterActivationError: '', clusterConnectionError: '',
  clusterDeactivatingId: null, clusterActivationLoading: false,
  clusterAutoconfigureLoading: false, clusterLinkSetupLoading: false,
  clusterQuickNodes: () => [{}, {}], clusterAllModels: () => [],
  clusterCatalogueFit: () => ({ fits: true }), clusterCatalogueLoading: false,
  clusterFabricLoading: false, clusterLinkStatusLoading: false,
  clusterPeerSsh: 'dk@peer', clusterDiscoveryLoading: false,
  clusterPeerProbeLoading: false, clusterPeerProbe: null,
  clusterSelectedModel: () => ({ model_path: 'mlx-community/model' }),
  clusterLiveJobs: () => [], clusterPrimaryDeployment: () => null,
  clusterFriendlyMacName: () => 'peer', clusterActivating: false,
  clusterLaunching: false, clusterPlanNodes: [],
  clusterModelInventory: {}, clusterModelInventoryLoading: false,
"""

_HEALTHY = "{ runtime_compatible: true, status: { node: { hostname: 'peer' } } }"
_MISMATCHED = (
    "{ runtime_compatible: false, bootstrap_required: false,"
    " runtime_mismatches: ['omlx local=1 remote=2'],"
    " status: { node: { hostname: 'peer' } } }"
)
# launch.py:1082 — oMLX absent or unverifiable on the peer, not a version skew.
_BOOTSTRAP = (
    "{ runtime_compatible: false, bootstrap_required: true,"
    " runtime_mismatches: ['oMLX worker runtime is not installed'],"
    " status: { node: { hostname: 'peer' } } }"
)
# A probe payload that predates the runtime fields entirely.
_NO_VERDICT = "{ status: { node: { hostname: 'peer' } } }"


def _statuses() -> dict[str, dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the dashboard status computation")

    methods = ",\n".join(_method_source(name) for name in _LIFTED)
    script = f"""
const component = {{
{_STUBS}
{methods}
}};
const out = {{}};
function sample(name) {{ out[name] = component.clusterQuickStatus(); }}

component.clusterPeerProbe = {_HEALTHY};
sample('settled');

component.clusterPeerProbeLoading = true;
sample('reprobing');

component.clusterPeerProbe = null;
component.clusterPeerProbeLoading = true;
sample('first_probe');
component.clusterPeerProbeLoading = false;

component.clusterPeerProbe = {_MISMATCHED};
sample('mismatched');

component.clusterConnectionError = 'peer is online, but its oMLX worker runtime is not installed yet.';
component.clusterPeerProbe = {_BOOTSTRAP};
sample('bootstrap');
component.clusterConnectionError = '';

component.clusterPeerProbe = {_NO_VERDICT};
sample('no_verdict');

console.log(JSON.stringify(out));
"""
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def statuses() -> dict[str, dict]:
    return _statuses()


def test_a_healthy_cluster_reports_ready(statuses):
    # Guards the stub set itself: if this stops being 'ready', the cases below
    # are comparing something other than the settled healthy state.
    assert statuses["settled"]["key"] == "ready"
    assert statuses["settled"]["busy"] is False


def test_the_first_probe_still_reports_checking(statuses):
    assert statuses["first_probe"]["key"] == "checking"
    assert statuses["first_probe"]["busy"] is True


def test_an_incompatible_runtime_is_a_verdict_not_a_loading_state(statuses):
    mismatched = statuses["mismatched"]

    assert mismatched["key"] == "runtime-mismatch"
    assert mismatched["busy"] is False
    assert mismatched["tone"] == "red"
    assert "omlx local=1 remote=2" in mismatched["detail"]


def test_a_worker_without_omlx_is_not_reported_as_a_version_mismatch(statuses):
    # bootstrap_required needs an install, not a version reconciliation.
    bootstrap = statuses["bootstrap"]

    assert bootstrap["key"] == "bootstrap"
    assert bootstrap["key"] != statuses["mismatched"]["key"]
    assert bootstrap["label"] == "Worker runtime setup needed"
    assert "not match" not in bootstrap["label"]
    assert "not installed" in bootstrap["detail"]
    assert bootstrap["busy"] is False


def test_a_probe_with_no_runtime_verdict_is_not_called_incompatible(statuses):
    # An older or partial payload must stay fail-closed without becoming red.
    unverified = statuses["no_verdict"]

    assert unverified["key"] == "runtime-unverified"
    assert unverified["tone"] == "amber"
    assert unverified["busy"] is False


def test_every_status_uses_a_tone_the_stylesheet_maps():
    tones = set(re.findall(r"tone: '(\w+)'", _method_source("clusterQuickStatus")))
    mapped = set(
        re.findall(r"^\s+(\w+): 'bg-", _method_source("clusterQuickStatusTone"), re.M)
    )

    assert tones <= mapped, f"unmapped tones would fall back to grey: {tones - mapped}"
