# SPDX-License-Identifier: Apache-2.0
"""Dashboard integration contracts for the sole Cluster v2 UI."""

import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from omlx.admin import routes as admin_routes

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "omlx/admin/templates/dashboard.html"
TEMPLATE = ROOT / "omlx/admin/templates/dashboard/_cluster_v2.html"
JAVASCRIPT = ROOT / "omlx/admin/static/js/cluster_v2.js"


def test_dashboard_renders_only_the_cluster_v2_flow():
    rendered = admin_routes.templates.get_template("dashboard.html").render()
    source = DASHBOARD.read_text(encoding="utf-8")

    assert "data-cluster-v2-wizard" in rendered
    assert source.count('{% include "dashboard/_cluster_v2.html" %}') == 1
    assert '_cluster.html' not in source
    assert "clusterLegacyView" not in source
    assert not (TEMPLATE.parent / "_cluster.html").exists()


def test_cluster_navigation_exists_for_desktop_and_mobile():
    navbar = (
        ROOT / "omlx/admin/templates/dashboard/_navbar.html"
    ).read_text(encoding="utf-8")

    assert navbar.count("setMainTab('cluster')") == 2
    assert navbar.count("mainTab === 'cluster'") == 3
    assert navbar.count("navbar.tab.cluster") == 2


def test_distributed_inference_remains_an_advanced_restart_scoped_opt_in():
    settings = (
        ROOT / "omlx/admin/templates/dashboard/_settings.html"
    ).read_text(encoding="utf-8")

    assert "settings.advanced.distributed_inference" in settings
    assert "settings.advanced.distributed_inference_enabled" in settings
    assert "settings.advanced.distributed_inference_hint" in settings


def test_v2_uses_authenticated_cluster_admin_apis():
    source = JAVASCRIPT.read_text(encoding="utf-8")

    for endpoint in (
        "/admin/api/cluster/models",
        "/admin/api/cluster/catalogue",
        "/admin/api/cluster/peer-probe",
        "/admin/api/cluster/autoconfigure",
        "/admin/api/cluster/runtime",
        "/admin/api/cluster/deployments",
        "/admin/api/cluster/replan",
    ):
        assert endpoint in source


def test_advanced_tools_preserve_cuda_connectx_and_diagnostics():
    template = TEMPLATE.read_text(encoding="utf-8")
    source = JAVASCRIPT.read_text(encoding="utf-8")

    for marker in (
        "data-cluster-v2-advanced-tools",
        "data-cluster-v2-cuda-enrollment",
        "data-cluster-v2-cuda-command",
        "data-cluster-v2-connectx-verify",
        "data-cluster-v2-diagnostics",
    ):
        assert marker in template
    for endpoint in (
        "/admin/api/cluster/join-keys",
        "/admin/api/cluster/join-status",
        "/admin/api/cluster/cuda-fabric/verify",
        "/admin/api/cluster/diagnostics",
    ):
        assert endpoint in source
    assert "Advanced (legacy)" not in template
    assert "clusterLegacyView" not in template + source


def test_dashboard_does_not_reference_an_unbundled_alpine_plugin():
    rendered = admin_routes.templates.get_template("dashboard.html").render()
    assert "@alpinejs/" not in rendered
    assert "alpine-collapse" not in rendered


def test_cluster_v2_template_tags_balance():
    void = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    class Balance(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.stack = []
            self.errors = []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append((tag, self.getpos()))

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack:
                self.errors.append(f"</{tag}> closes nothing")
                return
            opened, position = self.stack.pop()
            if opened != tag:
                self.errors.append(
                    f"</{tag}> closes <{opened}> opened at {position}"
                )

    parser = Balance()
    parser.feed(TEMPLATE.read_text(encoding="utf-8"))
    parser.close()
    assert not parser.errors
    assert parser.stack == []


def test_every_cluster_v2_alpine_expression_parses_as_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to parse Alpine expressions")
    source = TEMPLATE.read_text(encoding="utf-8")
    attribute = re.compile(
        r'(?P<name>(?:x-[a-z:.\-]+|@[A-Za-z0-9:.\-]+|:[A-Za-z0-9:.\-]+))'
        r'\s*=\s*"(?P<value>[^"]*)"',
        re.S,
    )
    statements = {"x-init", "x-data", "x-effect"}
    checks = []
    for match in attribute.finditer(source):
        name = match.group("name")
        if name == "x-cloak" or name.startswith("x-transition"):
            continue
        value = html.unescape(match.group("value")).strip()
        if value:
            base = name.split(".")[0].split(":")[0]
            checks.append(
                {
                    "name": name,
                    "line": source[: match.start()].count("\n") + 1,
                    "value": value,
                    "statement": base in statements or base.startswith("@"),
                }
            )
    assert len(checks) > 100
    script = """
const checks = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const failures = [];
for (const check of checks) {
  const body = check.statement ? check.value : `(${check.value})`;
  try {
    new Function('$event', '$el', '$refs', '$store', '$dispatch', '$nextTick', body);
  } catch (error) {
    failures.push(`${check.name} line ${check.line}: ${error.message}`);
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


def test_cluster_v2_javascript_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to parse cluster_v2.js")
    result = subprocess.run(
        [node, "--check", str(JAVASCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_every_dashboard_locale_names_cluster_tab():
    locale_dir = ROOT / "omlx/admin/i18n"
    for path in locale_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("navbar.tab.cluster"), path.name
