# SPDX-License-Identifier: Apache-2.0
"""B6 asset-version handshake: the stamp itself and where it may appear.

The stamp is a content hash of the shipped ``dashboard.js`` — deliberately
not an mtime, because the packaged app is staged with ditto/rsync, which
copy source mtimes (two different builds can collide; identical bytes can
look "new"). The header rides only on authenticated cluster API responses:
a login redirect must never be misread as version churn.
"""

import os
import time

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.cluster import routes as cluster_routes
from omlx.cluster.start_job import reset_start_job_store


@pytest.fixture(autouse=True)
def _fresh_state():
    """The version cache and job store are process-wide; isolate each test."""

    admin_routes._asset_version_cache.clear()
    reset_start_job_store()
    yield
    admin_routes._asset_version_cache.clear()
    reset_start_job_store()


def _static_tree(tmp_path, content: bytes):
    js_dir = tmp_path / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (js_dir / "dashboard.js").write_bytes(content)
    return tmp_path


def test_asset_version_tracks_content_not_mtime(tmp_path, monkeypatch):
    """Same bytes → same stamp even with a new mtime; new bytes → new stamp.

    This is the packaged-app property that matters: build staging may
    preserve or rewrite mtimes arbitrarily, and neither may fool the
    handshake.
    """

    monkeypatch.setattr(
        admin_routes, "static_dir", _static_tree(tmp_path, b"// build one\n")
    )
    first = admin_routes.asset_version()
    assert first and len(first) == 16 and set(first) <= set("0123456789abcdef")

    # Rewrite the identical bytes with a visibly different mtime.
    admin_routes._asset_version_cache.clear()
    path = tmp_path / "js" / "dashboard.js"
    path.write_bytes(b"// build one\n")
    os.utime(path, (time.time() + 3600, time.time() + 3600))
    assert admin_routes.asset_version() == first

    admin_routes._asset_version_cache.clear()
    path.write_bytes(b"// build two\n")
    assert admin_routes.asset_version() != first


def test_asset_version_is_cached_between_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(
        admin_routes, "static_dir", _static_tree(tmp_path, b"// cached\n")
    )
    first = admin_routes.asset_version()
    # A content change without a cache clear is invisible inside the TTL —
    # that is the point: no file read per request.
    (tmp_path / "js" / "dashboard.js").write_bytes(b"// changed\n")
    assert admin_routes.asset_version() == first


def test_dashboard_js_cache_buster_uses_the_same_content_hash(
    tmp_path, monkeypatch
):
    """The reload the stale bar demands must actually bust the browser cache."""

    monkeypatch.setattr(
        admin_routes, "static_dir", _static_tree(tmp_path, b"// bundle\n")
    )
    url = admin_routes._static_version("js/dashboard.js")
    assert url == f"/admin/static/js/dashboard.js?v={admin_routes.asset_version()}"


def _stamped_client(*extra_dependencies) -> TestClient:
    """An app wired like production: auth-style deps first, stamp last."""

    app = FastAPI()
    app.include_router(
        cluster_routes.router,
        dependencies=[
            *extra_dependencies,
            Depends(admin_routes.stamp_asset_version),
        ],
    )
    return TestClient(app)


def test_cluster_responses_carry_the_asset_version_header():
    response = _stamped_client().get("/admin/api/cluster/start-jobs")

    assert response.status_code == 200
    assert response.headers["X-Omlx-Asset-Version"] == admin_routes.asset_version()


def test_unauthenticated_responses_never_carry_the_header():
    """Auth runs first, so a 401 has no version header to misread (B6)."""

    def _deny() -> None:
        raise HTTPException(status_code=401, detail="not logged in")

    response = _stamped_client(Depends(_deny)).get(
        "/admin/api/cluster/start-jobs"
    )

    assert response.status_code == 401
    assert "X-Omlx-Asset-Version" not in response.headers


def test_missing_asset_degrades_to_no_header(tmp_path, monkeypatch):
    """No dashboard.js on disk → empty stamp → the handshake stands down."""

    monkeypatch.setattr(admin_routes, "static_dir", tmp_path)
    assert admin_routes.asset_version() == ""

    response = _stamped_client().get("/admin/api/cluster/start-jobs")

    assert response.status_code == 200
    assert "X-Omlx-Asset-Version" not in response.headers
