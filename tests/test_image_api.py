# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible image API (FLUX.2 Klein via mflux).

Fake engines stand in for mflux — no model download or diffusion run.
"""

import base64
import io
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from omlx.api.image_routes import router
from omlx.engine.image import REGISTRY, get_image_engine
from omlx.engine.image.flux2 import (
    Flux2KleinEditImageEngine,
    Flux2KleinImageEngine,
)
from omlx.model_discovery import (
    _is_mflux_image_model,
    _is_model_dir,
    detect_model_type,
)
from omlx.utils.mflux import resolve_mflux_config, resolve_mflux_family


def _png_bytes(size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


class FakeKleinModel:
    """Stands in for mflux Flux2Klein / Flux2KleinEdit."""

    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        result = MagicMock()
        result.image = Image.open(io.BytesIO(_png_bytes()))
        return result


def _engine(cls=Flux2KleinImageEngine, model_name="flux2-klein-9b"):
    engine = cls(model_name=model_name)
    engine._model = FakeKleinModel()
    return engine


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@contextmanager
def serving(engine):
    """Patch route deps: fake pool serving ``engine``, identity model ids."""
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=engine)
    with (
        patch("omlx.api.image_routes._get_engine_pool", return_value=pool),
        patch("omlx.api.image_routes._resolve_model", side_effect=lambda m: m),
        patch("omlx.api.image_routes.get_server_metrics"),
    ):
        yield engine


# ---------------------------------------------------------------------------
# Discovery & resolution
# ---------------------------------------------------------------------------


def test_mflux_component_dir_is_detected_as_image_model(tmp_path):
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "transformer").mkdir()
    assert _is_mflux_image_model(tmp_path)
    assert _is_model_dir(tmp_path)
    assert detect_model_type(tmp_path) == "image"


def test_llm_dir_not_image_model(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    assert not _is_mflux_image_model(tmp_path)
    assert detect_model_type(tmp_path) == "llm"


def test_model_index_json_counts_as_image_model(tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    assert _is_mflux_image_model(tmp_path)
    assert detect_model_type(tmp_path) == "image"


def test_family_resolution_txt2img_edit_and_reject():
    assert resolve_mflux_family(resolve_mflux_config("flux2-klein-9b")) is not None
    fam = resolve_mflux_family(resolve_mflux_config("flux2-klein-9b-edit"))
    assert fam.__name__ == "Flux2KleinEdit"
    with pytest.raises(ValueError, match="FLUX"):
        resolve_mflux_family(
            SimpleNamespace(model_name="Qwen/Qwen-Image-2512", aliases=[])
        )


def test_registry_maps_flux_families_only():
    assert set(REGISTRY) == {"Flux2Klein", "Flux2KleinEdit"}
    eng = get_image_engine("/models/flux2-klein-9b")
    assert isinstance(eng, Flux2KleinImageEngine)
    eng = get_image_engine("/models/flux2-klein-9b-edit")
    assert isinstance(eng, Flux2KleinEditImageEngine)


# ---------------------------------------------------------------------------
# Generations
# ---------------------------------------------------------------------------


def test_generation_returns_b64_png_and_seed(client):
    engine = _engine()
    with serving(engine):
        r = client.post(
            "/v1/images/generations",
            json={
                "model": "flux2-klein-9b",
                "prompt": "a cat",
                "seed": 7,
                "size": "1024x768",
            },
        )
    assert r.status_code == 200
    item = r.json()["data"][0]
    assert base64.b64decode(item["b64_json"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert item["seed"] == 7
    call = engine._model.calls[0]
    assert (call["width"], call["height"]) == (1024, 768)


def test_generation_n_seeds_increment_and_extra_params_forwarded(client):
    engine = _engine()
    with serving(engine):
        r = client.post(
            "/v1/images/generations",
            json={
                "model": "flux2-klein-9b",
                "prompt": "a dog",
                "n": 3,
                "seed": 100,
                "num_inference_steps": 28,
                "guidance": 2.5,
                "user": "junk-is-ignored",
            },
        )
    assert r.status_code == 200
    assert [d["seed"] for d in r.json()["data"]] == [100, 101, 102]
    assert all(
        c["num_inference_steps"] == 28 and c["guidance"] == 2.5
        for c in engine._model.calls
    )
    assert "user" not in engine._model.calls[0]


def test_generation_without_seed_reports_random_seeds(client):
    engine = _engine()
    with serving(engine):
        r = client.post(
            "/v1/images/generations",
            json={"model": "flux2-klein-9b", "prompt": "x"},
        )
    assert r.status_code == 200
    seeds = [d["seed"] for d in r.json()["data"]]
    assert len(seeds) == 1 and isinstance(seeds[0], int)


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m"},  # missing prompt
        {"model": "m", "prompt": "x", "num_inference_steps": 0},
        {"model": "m", "prompt": "x", "n": 9},
        {"model": "m", "prompt": "x", "width": 7},
    ],
)
def test_generation_validation_errors(client, body):
    engine = _engine()
    with serving(engine):
        r = client.post("/v1/images/generations", json=body)
    assert r.status_code == 422


def test_model_load_failure_is_404(client):
    pool = MagicMock()
    pool.get_engine = AsyncMock(side_effect=RuntimeError("no weights"))
    with (
        patch("omlx.api.image_routes._get_engine_pool", return_value=pool),
        patch("omlx.api.image_routes._resolve_model", side_effect=lambda m: m),
    ):
        r = client.post("/v1/images/generations", json={"model": "gone", "prompt": "x"})
    assert r.status_code == 404


def test_generation_failure_is_500(client):
    engine = _engine()
    engine._model.generate_image = MagicMock(side_effect=RuntimeError("Metal OOM"))
    with serving(engine):
        r = client.post(
            "/v1/images/generations", json={"model": "flux2-klein-9b", "prompt": "x"}
        )
    assert r.status_code == 500
    assert "Metal OOM" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------


def test_edit_json_resolves_data_uris_and_cleans_up(client):
    engine = _engine(Flux2KleinEditImageEngine)
    uris = [_data_uri(_png_bytes()), _data_uri(_png_bytes())]
    with serving(engine):
        r = client.post(
            "/v1/images/edits",
            json={
                "model": "flux2-klein-9b-edit",
                "prompt": "add a hat",
                "images": uris,
            },
        )
    assert r.status_code == 200
    assert base64.b64decode(r.json()["data"][0]["b64_json"])[:8] == b"\x89PNG\r\n\x1a\n"
    paths = engine._model.calls[0]["image_paths"]
    assert len(paths) == 2
    assert not any(Path(p).exists() for p in paths)  # temp files cleaned


def test_edit_json_accepts_image_url_objects_and_image_field(client):
    engine = _engine(Flux2KleinEditImageEngine)
    body = {
        "model": "flux2-klein-9b-edit",
        "prompt": "x",
        "images": [{"image_url": _data_uri(_png_bytes())}],
        "image_strength": 0.8,
    }
    with serving(engine):
        r = client.post("/v1/images/edits", json=body)
    assert r.status_code == 200
    call = engine._model.calls[0]
    assert call["image_strength"] == 0.8
    assert "image_path" not in call


def test_edit_klein_txt2img_engine_uses_single_image_path(client):
    engine = _engine(Flux2KleinImageEngine)
    body = {
        "model": "flux2-klein-9b",
        "prompt": "x",
        "images": [_data_uri(_png_bytes())],
    }
    with serving(engine):
        r = client.post("/v1/images/edits", json=body)
    assert r.status_code == 200
    call = engine._model.calls[0]
    assert "image_path" in call and "image_paths" not in call


def test_edit_requires_input_image(client):
    engine = _engine(Flux2KleinEditImageEngine)
    with serving(engine):
        r = client.post("/v1/images/edits", json={"model": "x", "prompt": "p"})
    assert r.status_code == 400


def test_edit_rejects_non_data_uri_images(client):
    engine = _engine(Flux2KleinEditImageEngine)
    body = {"model": "x", "prompt": "p", "images": ["https://example.com/a.png"]}
    with serving(engine):
        r = client.post("/v1/images/edits", json=body)
    assert r.status_code == 400


def test_edit_multipart_openai_wire_format(client):
    engine = _engine(Flux2KleinEditImageEngine)
    with serving(engine):
        r = client.post(
            "/v1/images/edits",
            data={
                "model": "flux2-klein-9b-edit",
                "prompt": "restyle",
                "n": "2",
                "seed": "5",
                "guidance": "3.5",
            },
            files=[("image", ("a.png", _png_bytes(), "image/png"))],
        )
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["seed"] for d in data] == [5, 6]
    call = engine._model.calls[0]
    assert call["guidance"] == 3.5
    assert call["image_strength"] == 0.5  # default applied


def test_edit_multipart_bracket_field_and_empty_prompt(client):
    engine = _engine(Flux2KleinEditImageEngine)
    with serving(engine):
        ok = client.post(
            "/v1/images/edits",
            data={"model": "m", "prompt": "p"},
            files=[("image[]", ("a.jpg", _png_bytes(), "image/jpeg"))],
        )
        bad = client.post(
            "/v1/images/edits",
            data={"model": "m", "prompt": "  "},
            files=[("image", ("a.png", _png_bytes(), "image/png"))],
        )
    assert ok.status_code == 200
    assert bad.status_code == 400


def test_edit_multipart_without_file_is_400(client):
    engine = _engine(Flux2KleinEditImageEngine)
    raw = (
        b'--X\r\nContent-Disposition: form-data; name="model"\r\n\r\nm\r\n'
        b'--X\r\nContent-Disposition: form-data; name="prompt"\r\n\r\np\r\n'
        b"--X--\r\n"
    )
    with serving(engine):
        r = client.post(
            "/v1/images/edits",
            content=raw,
            headers={"content-type": "multipart/form-data; boundary=X"},
        )
    assert r.status_code == 400
