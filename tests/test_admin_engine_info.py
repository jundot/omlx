# SPDX-License-Identifier: Apache-2.0
"""Admin engine metadata tests."""

import importlib.metadata

from omlx.admin import routes as admin_routes


def test_engine_info_omits_mlx_embeddings(monkeypatch):
    def distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(admin_routes, "_load_fallback_commits", lambda packages: {})

    engines = admin_routes._get_engine_info()

    assert "mlx-embeddings" not in engines
    assert "mlx-vlm" in engines
