# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI-compatible image routes."""

from __future__ import annotations

import asyncio
import base64
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from omlx.api import image_routes

RED, BLUE = (255, 0, 0), (0, 0, 255)


def _png_bytes(color: tuple[int, int, int] = RED) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class FakeImageEngine:
    def __init__(self) -> None:
        self.pool: FakePool | None = None
        self.generate_calls: list[dict] = []
        self.edit_calls: list[dict] = []
        self.seen_paths: list[str] = []

    async def generate(self, prompt: str, **kwargs):
        self.generate_calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(image=Image.new("RGB", (2, 2), (0, 255, 0)), metadata={})

    async def edit(self, prompt: str, image_paths: list[str], **kwargs):
        self.seen_paths.extend(image_paths)
        colors = []
        for path in image_paths:
            with Image.open(path) as image:
                colors.append(image.convert("RGB").getpixel((0, 0)))
        self.edit_calls.append({"prompt": prompt, "colors": colors, **kwargs})
        return SimpleNamespace(image=Image.new("RGB", (2, 2), BLUE), metadata={})


class FakePool:
    def __init__(self, engine, tasks, *, model_type="image", engine_type="image"):
        self.engine = engine
        engine.pool = self
        self.in_use = 0
        self.entry = SimpleNamespace(
            model_type=model_type, engine_type=engine_type, tasks=tasks
        )

    def get_entry(self, model_id: str):
        return self.entry if model_id == "resolved-image-model" else None

    async def get_engine(self, model_id: str, **_):
        self.in_use += 1
        return self.engine

    async def release_engine(self, model_id: str):
        self.in_use -= 1


@pytest.fixture
def image_client(monkeypatch):
    app = FastAPI()
    app.include_router(image_routes.router)
    monkeypatch.setattr(image_routes, "_resolve_model", lambda _: "resolved-image-model")
    monkeypatch.setattr(image_routes, "require_mlx_vlm_available", lambda: None)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _install_pool(monkeypatch, tasks: list[str], **entry) -> FakeImageEngine:
    engine = FakeImageEngine()
    pool = FakePool(engine, tasks, **entry)
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    return engine


def _post_edit(client, transport: str, inputs: list[bytes], **fields):
    if transport == "json":
        images = [{"image_url": _data_uri(data)} for data in inputs]
        return client.post(
            "/v1/images/edits",
            json={"model": "alias", "prompt": "edit it", "images": images, **fields},
        )
    return client.post(
        "/v1/images/edits",
        data={"model": "alias", "prompt": "edit it", **fields},
        files=[("image", (f"{i}.png", data, "image/png")) for i, data in enumerate(inputs)],
    )


def test_generation_returns_one_png_per_image_with_incrementing_seed(
    image_client, monkeypatch
):
    engine = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a green square", "n": 2, "seed": 41},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "resolved-image-model"
    assert len(body["data"]) == 2
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG")
    assert [call["seed"] for call in engine.generate_calls] == [41, 42]
    assert engine.pool.in_use == 0


@pytest.mark.parametrize("task", ["generation", "edit"])
@pytest.mark.parametrize("seed", [None, 0])
def test_batch_preserves_optional_seed(image_client, monkeypatch, task, seed):
    engine = _install_pool(monkeypatch, [task])
    fields = {"n": 2}
    if seed is not None:
        fields["seed"] = seed

    if task == "generation":
        response = image_client.post(
            "/v1/images/generations",
            json={"model": "alias", "prompt": "a square", **fields},
        )
        calls = engine.generate_calls
    else:
        response = _post_edit(image_client, "json", [_png_bytes()], **fields)
        calls = engine.edit_calls

    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    assert [call["seed"] for call in calls] == (
        [None, None] if seed is None else [seed, seed + 1]
    )


@pytest.mark.parametrize(
    "tasks,entry,detail",
    [
        (["generation"], {"model_type": "llm", "engine_type": "batched"}, "not an image model"),
        (["edit"], {}, "does not support task 'generation'"),
    ],
)
def test_generation_rejects_incompatible_models(
    image_client, monkeypatch, tasks, entry, detail
):
    engine = _install_pool(monkeypatch, tasks, **entry)

    response = image_client.post(
        "/v1/images/generations", json={"model": "alias", "prompt": "a square"}
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert engine.pool.in_use == 0


@pytest.mark.parametrize(
    "size,detail", [("wide", "WIDTHxHEIGHT"), ("2049x2048", "4194304")]
)
def test_generation_rejects_bad_or_oversized_output(
    image_client, monkeypatch, size, detail
):
    engine = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square", "size": size},
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert engine.generate_calls == []


@pytest.mark.parametrize("transport", ["json", "multipart"])
def test_edit_forwards_references_in_order_and_cleans_temp_files(
    image_client, monkeypatch, transport
):
    engine = _install_pool(monkeypatch, ["edit"])

    response = _post_edit(
        image_client, transport, [_png_bytes(RED), _png_bytes(BLUE)], seed="9"
    )

    assert response.status_code == 200
    assert base64.b64decode(response.json()["data"][0]["b64_json"]).startswith(
        b"\x89PNG"
    )
    assert engine.edit_calls[0]["colors"] == [RED, BLUE]
    assert engine.edit_calls[0]["seed"] == 9
    assert engine.seen_paths
    assert not any(Path(path).exists() for path in engine.seen_paths)
    assert engine.pool.in_use == 0


@pytest.mark.parametrize("transport", ["json", "multipart"])
@pytest.mark.parametrize(
    "limit,detail",
    [
        ("MAX_IMAGE_INPUT_BYTES", "exceeds maximum allowed size"),
        ("MAX_IMAGE_INPUT_PIXELS", "maximum allowed pixels"),
    ],
)
def test_edit_rejects_oversized_inputs(
    image_client, monkeypatch, transport, limit, detail
):
    engine = _install_pool(monkeypatch, ["edit"])
    monkeypatch.setattr(image_routes, limit, 1)

    response = _post_edit(image_client, transport, [_png_bytes()])

    assert response.status_code == 413
    assert detail in response.json()["detail"]
    assert engine.edit_calls == []


@pytest.mark.parametrize(
    "url_host,ip",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("169.254.169.254", "169.254.169.254"),
        ("100.64.0.1", "100.64.0.1"),
        ("[::1]", "::1"),
        ("mapped.test", "::ffff:10.0.0.1"),
    ],
)
def test_edit_rejects_non_public_image_urls(
    image_client, monkeypatch, url_host, ip
):
    engine = _install_pool(monkeypatch, ["edit"])
    monkeypatch.setattr(
        image_routes.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (image_routes.socket.AF_INET, image_routes.socket.SOCK_STREAM, 0, "", (ip, port))
        ],
    )

    response = _post_edit(
        image_client,
        "json",
        [],
        images=[{"image_url": f"http://{url_host}/image.png"}],
    )

    assert response.status_code == 400
    assert "public internet address" in response.json()["detail"]
    assert engine.edit_calls == []


PUBLIC_IP, INTERNAL_IP = "93.184.216.34", "10.0.0.5"


@pytest.fixture
def image_host(monkeypatch):
    """Serve images.example from a local server, faking only DNS and connect."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "http://internal.example/secret.png")
                self.end_headers()
                return
            body = _png_bytes(RED)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    dns = {"images.example": [PUBLIC_IP], "internal.example": [INTERNAL_IP]}
    connected: list[tuple[str, int]] = []
    create_connection, real_getaddrinfo = socket.create_connection, socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in dns:
            return real_getaddrinfo(host, port, *args, **kwargs)
        ip = dns[host].pop(0) if len(dns[host]) > 1 else dns[host][0]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, port))]

    def connect(address, *args, **kwargs):
        connected.append(address)
        if address[0] != PUBLIC_IP:
            raise ConnectionRefusedError(address)
        return create_connection(server.server_address, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", connect)
    yield SimpleNamespace(dns=dns, connected=connected)
    server.shutdown()
    server.server_close()


def test_edit_fetches_image_url_from_the_validated_address(
    image_client, monkeypatch, image_host
):
    engine = _install_pool(monkeypatch, ["edit"])
    image_host.dns["images.example"] = [PUBLIC_IP, INTERNAL_IP]

    response = _post_edit(
        image_client,
        "json",
        [],
        images=[{"image_url": "http://images.example/a.png"}],
    )

    assert response.status_code == 200, response.text
    assert engine.edit_calls[0]["colors"] == [RED]
    assert image_host.connected == [(PUBLIC_IP, 80)]


def test_edit_rejects_image_url_redirect_to_internal_host(
    image_client, monkeypatch, image_host
):
    engine = _install_pool(monkeypatch, ["edit"])

    response = _post_edit(
        image_client,
        "json",
        [],
        images=[{"image_url": "http://images.example/redirect"}],
    )

    assert response.status_code == 400
    assert "public internet address" in response.json()["detail"]
    assert image_host.connected == [(PUBLIC_IP, 80)]
    assert engine.edit_calls == []


async def test_cancelled_edit_keeps_inputs_and_pool_lease_until_worker_exits(
    monkeypatch,
):
    from dataclasses import replace

    from omlx.engine.image import ImageEngine, _image_api

    engine = ImageEngine(model_name="image-model", tasks=["edit"])
    engine._started = True
    pool = FakePool(engine, ["edit"])
    started, finish = threading.Event(), threading.Event()
    paths = []

    async def loaded(*args, **kwargs):
        return object()

    def worker(model, request, *, task):
        paths.extend(request.image_paths)
        started.set()
        assert finish.wait(5)
        assert all(Path(path).exists() for path in paths)
        return SimpleNamespace(image=Image.new("RGB", (2, 2)))

    monkeypatch.setattr(engine, "_load_model_for_task", loaded)
    api = replace(_image_api(), generate_image=worker)
    monkeypatch.setattr("omlx.engine.image._image_api", lambda: api)
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(image_routes, "_resolve_model", lambda _: "resolved-image-model")

    async def payload():
        return {
            "model": "image-model",
            "prompt": "change",
            "images": [{"image_url": _data_uri(_png_bytes())}],
        }

    task = asyncio.create_task(
        image_routes._create_json_image_edit(SimpleNamespace(json=payload))
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert pool.in_use == 1
        assert all(Path(path).exists() for path in paths)
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.in_use == 0
    assert not any(Path(path).exists() for path in paths)


def test_server_image_routes_inherit_api_key_auth(monkeypatch):
    from omlx import server

    engine = _install_pool(monkeypatch, ["generation"])
    monkeypatch.setattr(image_routes, "_resolve_model", lambda _: "resolved-image-model")
    monkeypatch.setattr(image_routes, "require_mlx_vlm_available", lambda: None)
    monkeypatch.setattr(server._server_state, "api_key", "secret")
    monkeypatch.setattr(server._server_state, "global_settings", None)
    monkeypatch.setattr(server._server_state, "engine_pool", None)
    monkeypatch.setattr(server._server_state, "process_memory_enforcer", None)
    body = {"model": "alias", "prompt": "a square"}

    with TestClient(server.app, raise_server_exceptions=False) as client:
        unauthorized = client.post("/v1/images/generations", json=body)
        authorized = client.post(
            "/v1/images/generations",
            json=body,
            headers={"Authorization": "Bearer secret"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert len(engine.generate_calls) == 1
