# SPDX-License-Identifier: Apache-2.0
"""The distributed surface remains dark until explicitly enabled."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omlx.cluster.exposure import distributed_inference_enabled

ROOT = Path(__file__).resolve().parents[1]


def _settings(enabled: bool):
    return SimpleNamespace(
        server=SimpleNamespace(distributed_inference_enabled=enabled)
    )


def test_distributed_inference_is_disabled_without_settings():
    assert distributed_inference_enabled(None) is False


def test_distributed_inference_requires_explicit_opt_in():
    assert distributed_inference_enabled(_settings(False)) is False
    assert distributed_inference_enabled(_settings(True)) is True


def test_server_uses_one_startup_snapshot_for_routes_and_bonjour():
    source = (ROOT / "omlx/server.py").read_text()

    assert (
        "_server_state.distributed_inference_enabled = is_enabled(global_settings)"
        in source
    )
    assert "if _server_state.distributed_inference_enabled:" in source
    assert "_register_cluster_routes()" in source
    assert "Depends(require_distributed_inference_enabled)" in source
    assert (
        "_server_state.global_settings is not None\n"
        "        and distributed_inference_enabled()"
    ) in source


def test_worker_join_routes_use_enrollment_auth_not_the_admin_cookie():
    source = (ROOT / "omlx/server.py").read_text()

    assert "from .cluster.routes import join_router as cluster_join_router" in source
    assert (
        "cluster_join_router,\n"
        "        dependencies=[Depends(require_distributed_inference_enabled)]"
    ) in source
    assert "configure_cluster_enrollment(base_path)" in source
