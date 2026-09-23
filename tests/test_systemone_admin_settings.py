# SPDX-License-Identifier: Apache-2.0
"""The System One switch and model picker on the admin global-settings surface.

The switch decides whether ``/jev`` answers, and the picker decides what
``jev-latest`` resolves to. Both travel through ``get_global_settings`` /
``update_global_settings``, which build one large dict — the readable-model list
reaches the pool through that function's **local** ``server_state``, so naming
the module global there is a ``NameError`` that takes the whole settings page
down. That is the regression this file pins.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from omlx.settings import GlobalSettings

# A real pool id, double-hyphen as the pool spells it.
CHECKPOINT_ID = "mlx-community--diffusiongemma-26B-A4B-it-4bit"


class _Pool:
    """Only ``systemone_model_ids`` is consulted; nothing is loaded."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self.id_calls = 0

    def systemone_model_ids(self) -> list[str]:
        self.id_calls += 1
        return list(self._ids)


class _State:
    def __init__(self, pool) -> None:
        self.engine_pool = pool


def _wire(monkeypatch, gs: GlobalSettings, pool=None):
    from omlx.admin import routes as admin_routes

    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: _State(pool))
    return admin_routes


class TestGetAdvertisesTheSwitch:
    def test_defaults_are_off_and_automatic(self, tmp_path, monkeypatch) -> None:
        gs = GlobalSettings(base_path=Path(tmp_path))
        assert gs.server.systemone_enabled is False
        assert gs.server.systemone_model == ""

        routes = _wire(monkeypatch, gs, _Pool(["a-diffusion"]))
        server = asyncio.run(routes.get_global_settings(is_admin=True))["server"]
        assert server["systemone_enabled"] is False
        assert server["systemone_model"] == ""

    def test_picker_options_are_the_readable_checkpoints_sorted(
        self, tmp_path, monkeypatch
    ) -> None:
        gs = GlobalSettings(base_path=Path(tmp_path))
        pool = _Pool(["z-diffusion", "a-diffusion"])
        routes = _wire(monkeypatch, gs, pool)

        server = asyncio.run(routes.get_global_settings(is_admin=True))["server"]
        assert server["systemone_readable_models"] == ["a-diffusion", "z-diffusion"]
        assert pool.id_calls == 1

    def test_no_pool_yet_still_answers_with_an_empty_picker(
        self, tmp_path, monkeypatch
    ) -> None:
        # A settings page that 500s before any model is discovered would make the
        # switch impossible to turn on, which is the chicken-and-egg case.
        gs = GlobalSettings(base_path=Path(tmp_path))
        routes = _wire(monkeypatch, gs, None)

        server = asyncio.run(routes.get_global_settings(is_admin=True))["server"]
        assert server["systemone_readable_models"] == []

    def test_pool_without_the_predicate_degrades_to_empty(
        self, tmp_path, monkeypatch
    ) -> None:
        gs = GlobalSettings(base_path=Path(tmp_path))
        routes = _wire(monkeypatch, gs, object())

        server = asyncio.run(routes.get_global_settings(is_admin=True))["server"]
        assert server["systemone_readable_models"] == []


class TestPostAppliesTheSwitch:
    def test_applies_live_and_persists(self, tmp_path, monkeypatch) -> None:
        # Imported first: loading omlx.server rebinds the admin getters, so this
        # file's _wire has to patch after it to win.
        from omlx import server as oserver

        gs = GlobalSettings(base_path=Path(tmp_path))
        monkeypatch.setattr(oserver._server_state, "systemone_enabled", False)
        monkeypatch.setattr(oserver._server_state, "systemone_model", "")
        routes = _wire(monkeypatch, gs, _Pool([]))
        request = routes.GlobalSettingsRequest(
            systemone_enabled=True,
            systemone_model=CHECKPOINT_ID,
        )

        result = asyncio.run(
            routes.update_global_settings(request=request, is_admin=True)
        )
        assert result["success"] is True
        assert gs.server.systemone_enabled is True
        assert gs.server.systemone_model == CHECKPOINT_ID
        # The /jev gate is a per-request dependency on _server_state, so the new
        # value reaching it is exactly what opens the surface without a restart.
        assert "systemone_enabled" in result["runtime_applied"]
        assert "systemone_model" in result["runtime_applied"]
        assert oserver._server_state.systemone_enabled is True
        assert oserver._server_state.systemone_model == CHECKPOINT_ID

        restored = GlobalSettings.load(base_path=Path(tmp_path))
        assert restored.server.systemone_enabled is True
        assert restored.server.systemone_model == CHECKPOINT_ID

    def test_blank_model_means_automatic_and_trims_whitespace(
        self, tmp_path, monkeypatch
    ) -> None:
        gs = GlobalSettings(base_path=Path(tmp_path))
        gs.server.systemone_model = "old-checkpoint"
        routes = _wire(monkeypatch, gs, _Pool([]))

        asyncio.run(
            routes.update_global_settings(
                request=routes.GlobalSettingsRequest(systemone_model="  "),
                is_admin=True,
            )
        )
        assert gs.server.systemone_model == ""

    def test_omitting_the_fields_keeps_current_values(
        self, tmp_path, monkeypatch
    ) -> None:
        gs = GlobalSettings(base_path=Path(tmp_path))
        gs.server.systemone_enabled = True
        gs.server.systemone_model = "keep-me"
        routes = _wire(monkeypatch, gs, _Pool([]))

        asyncio.run(
            routes.update_global_settings(
                request=routes.GlobalSettingsRequest(log_level="debug"), is_admin=True
            )
        )
        assert gs.server.systemone_enabled is True
        assert gs.server.systemone_model == "keep-me"
