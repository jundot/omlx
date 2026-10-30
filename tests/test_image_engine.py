# SPDX-License-Identifier: Apache-2.0
"""Tests for the mlx-vlm-backed image engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import omlx.engine.image as image_module
from omlx.engine.image import ImageEngine


class _FakeImageAPI:
    ImageGenerationRequest = ImageEditRequest = SimpleNamespace

    def __init__(self) -> None:
        self.loads: list[str] = []
        self.requests: list[tuple[str, SimpleNamespace]] = []

    def load_image_model(self, reference: str, *, task: str) -> str:
        self.loads.append(task)
        return task

    def generate_image(self, model: str, request, *, task: str) -> SimpleNamespace:
        self.requests.append((task, request))
        return SimpleNamespace(image=None, metadata={})


@pytest.fixture
def api(monkeypatch):
    api = _FakeImageAPI()
    monkeypatch.setattr(image_module, "_image_api", lambda: api)
    monkeypatch.setattr(image_module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(image_module.mx, "clear_cache", lambda: None)
    return api


async def test_edit_forwards_inputs_and_strength_for_every_family(api):
    engine = ImageEngine(model_name="edit-model", tasks=["edit"])
    await engine.start()

    await engine.edit(
        "make it cinematic",
        image_paths=["input-a.png", "input-b.png"],
        seed=123,
        image_strength=0.55,
    )

    task, request = api.requests[-1]
    assert task == "edit"
    assert request.image_paths == ("input-a.png", "input-b.png")
    assert request.seed == 123
    assert request.extra["image_strength"] == request.extra["strength"] == 0.55


async def test_dual_task_engine_swaps_variants_instead_of_holding_both(api):
    engine = ImageEngine(model_name="dual-task", tasks=["generation", "edit"])
    await engine.start()

    await engine.generate("a cat")
    await engine.edit("make it brighter", image_paths=["input.png"])
    assert engine.get_stats()["loaded_tasks"] == ["edit"]
    await engine.generate("a dog")

    assert engine.get_stats()["loaded_tasks"] == ["generation"]
    assert api.loads == ["generate", "edit", "generate"]
