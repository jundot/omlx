# SPDX-License-Identifier: Apache-2.0
"""The admin API lists each loaded model that prefills remotely, and the dashboard has somewhere to show it."""

from __future__ import annotations

from types import SimpleNamespace

from omlx.admin import routes as admin_routes
from omlx.remote_prefill.routes import remote_prefill_status


def _engine(status):
    scheduler = SimpleNamespace(remote_prefill_status=lambda: status)
    return SimpleNamespace(
        _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    )


class _Pool:
    def __init__(self, entries):
        self._entries = entries

    def get_loaded_model_ids(self):
        return list(self._entries)

    def get_entry(self, model_id):
        return self._entries[model_id]


def test_only_models_that_prefill_remotely_are_listed():
    state = {"url": "http://prefill-host:8000", "links": ["sparka"], "paused": False}
    pool = _Pool(
        {
            "remote-model": SimpleNamespace(engine=_engine(state)),
            "local-model": SimpleNamespace(engine=_engine(None)),
            "distributed": SimpleNamespace(engine=SimpleNamespace()),
        }
    )
    assert remote_prefill_status(pool) == {
        "models": [{"model_id": "remote-model", **state}]
    }


def test_the_dashboard_has_a_remote_prefill_card():
    rendered = admin_routes.templates.get_template("dashboard.html").render()
    assert "data-cluster-v2-remote-prefill" in rendered
