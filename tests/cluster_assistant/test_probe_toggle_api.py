# SPDX-License-Identifier: Apache-2.0
"""B3: the in-app performance-probe switch."""

from __future__ import annotations

from assistant_harness import assistant_api

PROBE_URL = "/admin/api/cluster/assistant/performance-probe"


def test_the_probe_switch_round_trips_and_defaults_off(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, _store, _ops):
        first = client.get(PROBE_URL)
        assert first.status_code == 200
        assert first.json() == {"enabled": False}

        enable = client.post(PROBE_URL, json={"enabled": True})
        assert enable.status_code == 200
        assert enable.json() == {"enabled": True}
        assert client.get(PROBE_URL).json() == {"enabled": True}

        disable = client.post(PROBE_URL, json={"enabled": False})
        assert disable.status_code == 200
        assert disable.json() == {"enabled": False}


def test_the_probe_switch_refuses_unknown_fields(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, _store, _ops):
        response = client.post(PROBE_URL, json={"enabled": True, "restart": True})

    assert response.status_code == 422
