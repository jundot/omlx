# SPDX-License-Identifier: Apache-2.0
"""Per-request client attribution: key identity, IP fallback, stream propagation."""

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from omlx.admin.auth import fingerprint_key, identify_api_key, verify_any_api_key
from omlx.client_identity import (
    ClientIdentityMiddleware,
    _peer_label,
    current_client,
    set_key_identity,
)
from omlx.server_metrics import ServerMetrics
from omlx.usage_history import UsageHistory


def _sub(key, name=""):
    return SimpleNamespace(key=key, name=name)


def test_identify_api_key_reports_matching_key_without_secret():
    subs = [_sub("sub-a", "Home Assistant"), _sub("sub-b", "  "), _sub("", "blank")]
    assert identify_api_key("main", "main", subs) == ("main_key", "")
    assert identify_api_key("sub-a", "main", subs) == ("sub_key", "Home Assistant")
    # Unnamed keys fall back to the same fingerprint used for rejected-key logs.
    assert identify_api_key("sub-b", "main", subs) == (
        "sub_key",
        fingerprint_key("sub-b"),
    )
    assert identify_api_key("wrong", "main", subs) is None
    assert identify_api_key("", "main", subs) is None
    assert verify_any_api_key("sub-a", "main", subs) is True
    assert verify_any_api_key("wrong", "main", subs) is False


@pytest.mark.parametrize(
    ("client", "label"),
    [
        (("192.168.1.20", 5000), "192.168.1.20"),
        (("::ffff:10.0.0.7", 5000), "10.0.0.7"),
        (("::1", 5000), "::1"),
        (("testclient", 50000), "testclient"),
        (None, "unknown"),
    ],
)
def test_peer_label(client, label):
    assert _peer_label({"client": client}) == label


def test_identity_is_none_outside_requests():
    set_key_identity("sub_key", "ignored")
    assert current_client() is None


@pytest.fixture
def configured_server(tmp_path):
    from omlx import server

    history = UsageHistory(tmp_path / "usage.sqlite3", by_client=True)
    metrics = ServerMetrics()
    metrics.usage_history = history
    settings = SimpleNamespace(
        auth=SimpleNamespace(
            sub_keys=[_sub("editor-key", "Editor")],
            skip_api_key_verification=False,
            allow_unauthenticated_inference=False,
        ),
        server=SimpleNamespace(host="127.0.0.1"),
    )
    saved = (server._server_state.api_key, server._server_state.global_settings)
    server._server_state.api_key = "main-key"
    server._server_state.global_settings = settings

    app = FastAPI()
    app.add_middleware(ClientIdentityMiddleware)

    @app.get("/plain", dependencies=[Depends(server.verify_api_key)])
    async def plain():
        metrics.record_request_complete(10, 2, 0, 0.1, 0.1, "model-a", 0.2)
        return {"client": current_client()}

    @app.get("/stream", dependencies=[Depends(server.verify_api_key)])
    async def stream():
        async def body():
            yield "chunk"
            # Accounting runs inside the streamed body, after the handler returned.
            metrics.record_request_complete(5, 1, 0, 0.1, 0.1, "model-a", 0.2)

        return StreamingResponse(body())

    @app.get("/open")
    async def open_route():
        metrics.record_request_complete(1, 1, 0, 0.1, 0.1, "model-a", 0.2)
        return {"client": current_client()}

    @app.websocket("/ws")
    async def websocket_route(websocket: WebSocket):
        from omlx.api.audio_routes import _verify_ws_api_key

        if not _verify_ws_api_key(websocket, websocket.query_params.get("key")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        metrics.record_request_complete(3, 1, 0, 0.1, 0.1, "model-a", 0.2)
        await websocket.send_json({"client": current_client()})
        await websocket.close()

    try:
        yield TestClient(app, client=("192.168.1.20", 5000)), history
    finally:
        server._server_state.api_key, server._server_state.global_settings = saved
        history.close()


def test_requests_record_key_and_peer(configured_server):
    client, history = configured_server
    auth = {"Authorization": "Bearer editor-key"}
    peer = "192.168.1.20"
    assert client.get("/plain", headers=auth).json()["client"] == [
        "sub_key",
        "Editor",
        peer,
    ]
    assert client.get("/plain", headers={"x-api-key": "main-key"}).json()["client"] == [
        "main_key",
        "",
        peer,
    ]
    assert client.get("/open").json()["client"] == ["none", "", peer]
    assert client.get("/plain", headers={"x-api-key": "nope"}).status_code == 401
    assert client.get("/stream", headers=auth).text == "chunk"

    history.flush()
    result = history.query("today")
    keys = {(r["key_kind"], r["key_id"]): r for r in result["clients_by_key"]}
    assert keys[("sub_key", "Editor")]["requests"] == 2
    assert keys[("sub_key", "Editor")]["prompt_tokens"] == 15
    assert keys[("main_key", "")]["requests"] == 1
    assert keys[("none", "")]["requests"] == 1
    # The rejected request never reached accounting.
    assert [(r["client_ip"], r["requests"]) for r in result["clients_by_ip"]] == [
        (peer, 4)
    ]


def test_websocket_records_matching_key_and_peer(configured_server):
    from starlette.websockets import WebSocketDisconnect

    client, history = configured_server
    with client.websocket_connect("/ws?key=editor-key") as websocket:
        assert websocket.receive_json()["client"] == [
            "sub_key",
            "Editor",
            "192.168.1.20",
        ]
    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect("/ws?key=wrong"):
            pytest.fail("invalid key accepted")
    assert rejected.value.code == 1008
    history.flush()
    rows = history.query("today")["clients_by_key"]
    assert [(row["key_id"], row["requests"]) for row in rows] == [("Editor", 1)]


def test_locked_client_is_rejected_before_key_attribution(
    configured_server, monkeypatch
):
    from omlx.admin import auth

    client, history = configured_server
    monkeypatch.setattr(
        auth,
        "_key_failures",
        {"192.168.1.20": auth._KeyFailures(locked_until=auth.time.monotonic() + 300)},
    )
    response = client.get("/plain", headers={"x-api-key": "editor-key"})
    assert response.status_code == 429
    assert history.query("today")["clients"] == []
