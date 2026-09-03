# SPDX-License-Identifier: Apache-2.0
"""Persistence, live-apply, and UI contract for latent Metal keepwarm."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import omlx.server
from omlx.admin import routes
from omlx.engine_pool import EnginePool
from omlx.settings import GlobalSettings

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_TEMPLATE = ROOT / "omlx/admin/templates/dashboard/_settings.html"
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"
I18N_DIR = ROOT / "omlx/admin/i18n"


def test_admin_toggle_persists_and_hot_applies_loaded_and_future_engines(
    tmp_path,
    monkeypatch,
):
    settings = GlobalSettings(base_path=tmp_path)
    core = SimpleNamespace(configure_keepwarm=MagicMock())
    entry = SimpleNamespace(
        engine=SimpleNamespace(_engine=SimpleNamespace(engine=core))
    )
    configure_pool = MagicMock(
        side_effect=lambda enabled: core.configure_keepwarm(enabled)
    )
    pool = SimpleNamespace(
        _entries={"model": entry},
        configure_latent_metal_keepwarm=configure_pool,
    )
    monkeypatch.setattr(routes, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(omlx.server._server_state, "engine_pool", pool)
    monkeypatch.delenv("OMLX_KEEPWARM", raising=False)
    monkeypatch.delenv("OMLX_KEEPWARM_PROMPT_TAIL", raising=False)

    result = asyncio.run(
        routes.update_global_settings(
            request=routes.GlobalSettingsRequest(latent_metal_keepwarm_enabled=True),
            is_admin=True,
        )
    )

    assert result["success"] is True
    assert "latent_metal_keepwarm_enabled" in result["runtime_applied"]
    assert settings.server.latent_metal_keepwarm_enabled is True
    configure_pool.assert_called_once_with(True)
    assert core.configure_keepwarm.call_args.args == (True,)
    assert os.environ["OMLX_KEEPWARM"] == "1"
    assert os.environ["OMLX_KEEPWARM_PROMPT_TAIL"] == "1"
    persisted = json.loads((tmp_path / "settings.json").read_text())
    assert persisted["server"]["latent_metal_keepwarm_enabled"] is True


def test_pool_policy_reconciles_an_engine_published_after_live_toggle():
    existing_core = SimpleNamespace(configure_keepwarm=MagicMock())
    future_core = SimpleNamespace(configure_keepwarm=MagicMock())
    existing = SimpleNamespace(
        engine=SimpleNamespace(_engine=SimpleNamespace(engine=existing_core))
    )
    future = SimpleNamespace(engine=None)
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"existing": existing, "future": future}
    pool._latent_metal_keepwarm_enabled = False

    pool.configure_latent_metal_keepwarm(True)
    existing_core.configure_keepwarm.assert_called_once_with(True)

    # Simulate _load_engine publishing after the settings traversal completed.
    future.engine = SimpleNamespace(_engine=SimpleNamespace(engine=future_core))
    pool._apply_latent_metal_keepwarm_to_engine(future.engine)
    future_core.configure_keepwarm.assert_called_once_with(True)


def test_direct_server_init_seeds_future_engine_policy(tmp_path, monkeypatch):
    settings = GlobalSettings(base_path=tmp_path / "base")
    settings.server.latent_metal_keepwarm_enabled = True
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.delenv("OMLX_KEEPWARM", raising=False)
    # FastAPI forbids adding middleware once its stack has been built by an
    # earlier test; init_server's established tests reset it the same way.
    omlx.server.app.middleware_stack = None

    omlx.server.init_server(
        model_dirs=[str(model_dir)],
        global_settings=settings,
    )

    assert os.environ["OMLX_KEEPWARM"] == "1"
    assert omlx.server._server_state.engine_pool._latent_metal_keepwarm_enabled is True


def test_advanced_ui_exposes_default_off_experimental_toggle():
    template = SETTINGS_TEMPLATE.read_text(encoding="utf-8")
    javascript = DASHBOARD_JS.read_text(encoding="utf-8")

    performance = template.index("<!-- Performance subsection -->")
    keepwarm = template.index("settings.advanced.latent_metal_keepwarm")
    cache_section = template.index("<!-- Cache subsection -->")
    assert performance < keepwarm < cache_section
    assert "latent_metal_keepwarm_enabled: false" in javascript
    assert (
        "latent_metal_keepwarm_enabled: this.globalSettings.server."
        "latent_metal_keepwarm_enabled" in javascript
    )


def test_keepwarm_ui_keys_exist_in_every_locale():
    required = {
        "settings.advanced.experimental_badge",
        "settings.advanced.latent_metal_keepwarm",
        "settings.advanced.latent_metal_keepwarm_hint",
    }
    for locale_path in sorted(I18N_DIR.glob("*.json")):
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        assert not (required - set(locale)), locale_path.name
